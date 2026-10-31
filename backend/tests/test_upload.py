import io
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from app.main import app
from app.algorithms.schedule_extractor import parse_schedule_from_text, extract_operations_sequence

client = TestClient(app)

def get_test_font(size: int = 24):
    for font_path in [
        "/usr/share/fonts/ttf-readex-pro/ReadexPro-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "DejaVuSans.ttf"
    ]:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()

def create_schedule_image(text: str, fmt: str = "PNG") -> bytes:
    """Helper to generate an image buffer with rendered text."""
    font = get_test_font(24)
    img = Image.new("RGB", (700, 140), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 40), text, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def test_extract_operations_sequence_standard():
    text = "Consider schedule S = R1(X), W2(X), R1(Y), W2(Y). Test serializability."
    ops = extract_operations_sequence(text)
    assert ops == ["R1(X)", "W2(X)", "R1(Y)", "W2(Y)"]

def test_extract_operations_sequence_subscripts_and_lowercase():
    text = "r1(x), w2(x), r1(y), w2(y)"
    res = parse_schedule_from_text(text)
    assert res["parsed_schedule"] == "R1(X), W2(X), R1(Y), W2(Y)"
    assert res["is_valid"] is True
    assert res["operation_count"] == 4

def test_extract_operations_colon_notation():
    text = "T1: R(X), T2: W(X), T1: W(X), T3: W(X)"
    res = parse_schedule_from_text(text)
    assert res["parsed_schedule"] == "R1(X), W2(X), W1(X), W3(X)"
    assert res["is_valid"] is True

def test_extract_bracket_notation():
    text = "R1[X], W2[X], R1[Y], W2[Y]"
    res = parse_schedule_from_text(text)
    assert res["parsed_schedule"] == "R1(X), W2(X), R1(Y), W2(Y)"
    assert res["is_valid"] is True

def test_extract_incomplete_uncertain_tokens():
    text = "R1(X), W(X), R1(Y)"
    res = parse_schedule_from_text(text)
    assert "R1(X)" in res["parsed_schedule"]
    assert "R1(Y)" in res["parsed_schedule"]
    assert any("missing transaction id" in u.lower() for u in res["uncertain_parts"])

def test_extract_unrelated_text():
    text = "Welcome to the Database Management course midterm exam."
    res = parse_schedule_from_text(text)
    assert res["parsed_schedule"] == ""
    assert res["operation_count"] == 0
    assert res["is_valid"] is False
    assert len(res["validation_errors"]) > 0

def test_upload_invalid_extension():
    response = client.post(
        "/api/upload/question",
        files={"file": ("test.docx", b"R1(X), W2(X)", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_upload_empty_file():
    response = client.post(
        "/api/upload/question",
        files={"file": ("test.png", b"", "image/png")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]

def test_upload_file_exceeds_size_limit():
    large_payload = b"A" * (10 * 1024 * 1024 + 10)
    response = client.post(
        "/api/upload/question",
        files={"file": ("large.txt", large_payload, "text/plain")}
    )
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"].lower()


def test_upload_png_schedule_ocr():
    img_bytes = create_schedule_image("R1(X), W2(X), R3(Y), W1(Y)", fmt="PNG")
    response = client.post(
        "/api/upload/question",
        files={"file": ("question.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["file_type"] == "image"
    assert "R1(X)" in data["parsed_schedule"]
    assert "W2(X)" in data["parsed_schedule"]
    assert data["is_valid"] is True
    assert data["operation_count"] >= 3

def test_upload_webp_schedule_ocr():
    img_bytes = create_schedule_image("R1(X), W2(X), W1(X), W3(X)", fmt="WEBP")
    response = client.post(
        "/api/upload/question",
        files={"file": ("question.webp", img_bytes, "image/webp")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["file_type"] == "image"
    assert "R1(X)" in data["parsed_schedule"]
    assert data["is_valid"] is True

def test_upload_jpg_schedule_ocr():
    img_bytes = create_schedule_image("R1(X), W2(X), R1(Y), W2(Y)", fmt="JPEG")
    response = client.post(
        "/api/upload/question",
        files={"file": ("question.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "R1(X)" in data["parsed_schedule"]
    assert data["is_valid"] is True

def test_upload_unrelated_image_ocr():
    img_bytes = create_schedule_image("Database Systems chapter 5 notes without operations", fmt="PNG")
    response = client.post(
        "/api/upload/question",
        files={"file": ("notes.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["parsed_schedule"] == ""
    assert data["operation_count"] == 0
    assert data["is_valid"] is False
    assert len(data["validation_errors"]) > 0

def test_upload_pdf_digital_text():
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 700, "S = R1(X), W2(X), R1(Y), W2(Y)")
    c.save()
    pdf_bytes = buf.getvalue()

    response = client.post(
        "/api/upload/question",
        files={"file": ("exam.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["file_type"] == "pdf"
    assert "R1(X)" in data["parsed_schedule"]
    assert data["is_valid"] is True
