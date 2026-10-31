import os
import base64
import io
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pypdf import PdfReader
from PIL import Image
import pytesseract
from app.algorithms.schedule_extractor import (
    parse_schedule_from_text,
    extract_table_schedule_from_boxes,
    deduplicate_ocr_ops,
)
from app.algorithms.image_preprocessor import (
    prepare_image_variants,
    normalize_image_mode,
)
import httpx

# Auto-detect Tesseract OCR binary path across Linux / Docker / local environments
tesseract_bin = shutil.which("tesseract") or "/usr/bin/tesseract"
if os.path.isfile(tesseract_bin):
    pytesseract.pytesseract.tesseract_cmd = tesseract_bin

router = APIRouter(prefix="/api/upload", tags=["Document & Question Upload"])

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".pdf", ".txt"}


def preprocess_image(file_bytes: bytes) -> Image.Image:
    """
    Backwards-compatible helper returning a clean non-destructively preprocessed PIL Image.
    """
    try:
        base = Image.open(io.BytesIO(file_bytes))
        return normalize_image_mode(base)
    except Exception:
        return Image.new('RGB', (100, 100), (255, 255, 255))


def extract_text_from_image_bytes(file_bytes: bytes) -> str:
    """
    High-accuracy multi-candidate OCR extraction.
    Evaluates candidates from non-destructive image preprocessing variants (raw, grayscale, scaled, mild contrast)
    using schedule-aware validation. Selects the candidate that yields the highest-coherence schedule.
    """
    try:
        variants = prepare_image_variants(file_bytes)
        if not variants:
            return ""

        best_text = ""
        best_score = -1.0
        var_dict = dict(variants)
        priority_order = ['original', 'grayscale', 'mild_contrast', 'scaled', 'autocontrast']

        for v_name in priority_order:
            if v_name not in var_dict:
                continue
            var_img = var_dict[v_name]
            for psm in (6, 3):
                try:
                    txt = pytesseract.image_to_string(var_img, config=f'--psm {psm}').strip()
                    if not txt:
                        continue
                    parsed = parse_schedule_from_text(txt)
                    score = parsed["confidence_score"] * (parsed["operation_count"] + 1)
                    has_labeled = any(c.get("source_type") == "labeled" for c in parsed["candidate_schedules"])
                    if has_labeled:
                        score += 2.0

                    if score > best_score:
                        best_score = score
                        best_text = txt

                    # Fast exit: if exact match on a labeled schedule with >= 2 operations, return immediately
                    if parsed["status"] == "exact_match" and parsed["operation_count"] >= 2 and has_labeled:
                        return best_text
                except Exception:
                    pass

        # Fallback if no text extracted through variants
        if not best_text and variants:
            best_text = pytesseract.image_to_string(variants[0][1], config='--psm 6').strip()

        return best_text
    except Exception as e:
        print(f"Local OCR extraction error: {e}")
        return ""


@router.post("/question")
async def upload_question_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded PNG, JPG, JPEG, WEBP, or PDF question document.
    Extracts question text using multi-candidate local OCR (pytesseract) or PDF text extraction,
    with fallback to AI Vision (if GEMINI_API_KEY configured).
    Parses transaction operations into standard DBMS format: R1(X), W2(X), ...
    Returns structured confidence, candidate schedules, and validation results.
    """
    filename = file.filename or "uploaded_file"
    ext = os.path.splitext(filename.lower())[1]

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: PNG, JPG, JPEG, WEBP, PDF, TXT."
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_upload_bytes = 10 * 1024 * 1024  # 10 MB limit
    if len(file_bytes) > max_upload_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds the maximum allowed size of 10 MB.")

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    ai_available = bool(gemini_key)
    ai_used = False
    extracted_text = ""
    message = ""
    ocr_succeeded = False

    # Case A: Plain Text File (.txt)
    if ext == ".txt":
        try:
            extracted_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            extracted_text = file_bytes.decode("latin-1", errors="replace")
        extracted_text = extracted_text.strip()
        ocr_succeeded = True
        message = "Loaded schedule from text file."

    # Case B: PDF Document
    elif ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            extracted_text = "\n\n".join(pages_text).strip()

            parsed_initial = parse_schedule_from_text(extracted_text)
            if parsed_initial["operation_count"] == 0:
                ocr_texts = []
                for page in reader.pages:
                    for img_obj in page.images:
                        t = extract_text_from_image_bytes(img_obj.data)
                        if t:
                            ocr_texts.append(t)
                if ocr_texts:
                    extracted_text = "\n\n".join(ocr_texts).strip()
                    ocr_succeeded = True
                    message = "Extracted text and schedule from scanned PDF images using local OCR."
                elif extracted_text:
                    ocr_succeeded = True
                    message = "Extracted text from PDF document."
            else:
                ocr_succeeded = True
                message = "Extracted schedule successfully from PDF document."
        except Exception as e:
            extracted_text = ""
            message = f"Notice reading PDF: {str(e)}"

        # If PDF was scanned and local OCR found nothing, but AI is available, try Gemini
        parsed_pdf = parse_schedule_from_text(extracted_text)
        if parsed_pdf["operation_count"] == 0 and ai_available:
            ai_res = await call_gemini_vision(file_bytes, "application/pdf", gemini_key)
            if ai_res["success"] and ai_res["text"]:
                extracted_text = ai_res["text"]
                ai_used = True
                ocr_succeeded = True
                message = "Extracted text and schedule using AI Document Vision (Gemini)."

    # Case B: Image (PNG, JPG, JPEG, WEBP)
    else:
        # Run local OCR first with multi-candidate evaluation
        extracted_text = extract_text_from_image_bytes(file_bytes)

        if extracted_text.strip():
            ocr_succeeded = True

        # Also attempt bounding-box based table extraction directly from image
        table_ops_from_boxes = extract_table_schedule_from_boxes(file_bytes)
        parsed_temp = parse_schedule_from_text(extracted_text)

        # If bounding-box table extraction found operations, prioritize its visual column layout
        if table_ops_from_boxes and len(table_ops_from_boxes) >= parsed_temp["operation_count"]:
            table_text = ", ".join(table_ops_from_boxes)
            if extracted_text:
                extracted_text = f"Schedule S: {table_text}\n\n{extracted_text}"
            else:
                extracted_text = f"Schedule S: {table_text}"
            parsed_temp = parse_schedule_from_text(extracted_text)

        if parsed_temp["status"] in ("exact_match", "high_confidence"):
            message = "Schedule successfully recognized and extracted using local OCR."
        elif parsed_temp["status"] == "needs_review":
            message = "Schedule extracted with warnings. Please review the highlighted operations before analysis."
        elif ai_available:
            # Fall back to AI vision if local OCR didn't find operations
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp"
            }
            mime_type = mime_map.get(ext, "image/png")
            ai_res = await call_gemini_vision(file_bytes, mime_type, gemini_key)
            if ai_res["success"] and ai_res["text"]:
                extracted_text = ai_res["text"]
                ai_used = True
                ocr_succeeded = True
                message = "Extracted text and schedule from image using AI Vision (Gemini)."
            else:
                if ocr_succeeded:
                    message = "Text was recognized from the image, but no valid DBMS transaction operations were found."
                else:
                    message = "No text could be recognized from the image. AI Vision also could not read the schedule."
        else:
            if ocr_succeeded:
                message = "Text was recognized from the image, but no valid DBMS transaction operations were found."
            else:
                message = "No text could be recognized from the image. Please upload a clear photo or screenshot."

    # Parse and validate schedule from extracted text using deterministic extractor & existing analyzer validator
    parse_result = parse_schedule_from_text(extracted_text)

    if parse_result["operation_count"] > 0:
        if not message:
            message = f"Found {parse_result['operation_count']} transaction operations in schedule."
    else:
        if not message:
            if ocr_succeeded:
                message = "Text was detected, but no valid transaction schedule was found. Please check the raw text below."
            else:
                message = "No recognizable transaction operations found in the question."

    return {
        "success": True,
        "filename": filename,
        "file_type": "text" if ext == ".txt" else ("pdf" if ext == ".pdf" else "image"),
        "raw_text": parse_result["raw_text"],
        "parsed_schedule": parse_result["parsed_schedule"],
        "candidate_schedules": parse_result["candidate_schedules"],
        "operation_count": parse_result["operation_count"],
        "uncertain_parts": parse_result["uncertain_parts"],
        "is_valid": parse_result["is_valid"],
        "validation_errors": parse_result["validation_errors"],
        "validation_warnings": parse_result["validation_warnings"],
        "confidence_score": parse_result.get("confidence_score", 0.0),
        "status": parse_result.get("status", "ocr_error"),
        "warnings": parse_result.get("warnings", []),
        "selection_reason": parse_result.get("selection_reason", ""),
        "ai_available": ai_available,
        "ai_used": ai_used,
        "message": message,
        "ocr_succeeded": ocr_succeeded
    }


async def call_gemini_vision(file_bytes: bytes, mime_type: str, api_key: str) -> dict:
    """Safely calls Google Gemini Vision API as an AI enhancement when configured."""
    b64_data = base64.b64encode(file_bytes).decode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    prompt = (
        "You are an expert OCR and DBMS schedule extractor. Extract the database transaction schedule from this image/document. "
        "Transcribe only the operations in standard DBMS notation: e.g. R1(X), W2(X), R3(Y), W1(X), W2(Y), R1(Y), W3(X). "
        "Preserve exact chronological order. Output the schedule clearly as: Schedule: <comma-separated operations>."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data
                        }
                    }
                ]
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return {"success": True, "text": parts[0].get("text", "")}
            return {"success": False, "text": "", "error": f"API error: {resp.status_code}"}
    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}
