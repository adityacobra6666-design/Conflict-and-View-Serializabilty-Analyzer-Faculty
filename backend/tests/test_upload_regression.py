"""
Comprehensive regression tests for the Upload Question feature.
Tests cover: inline schedules, table schedules, bounding box extraction,
natural language parsing, edge cases, ambiguous ordering, and the full
upload-to-analysis pipeline.
"""
import io
import os
import re
from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient
from app.main import app
from app.algorithms.schedule_extractor import (
    parse_schedule_from_text,
    extract_operations_sequence,
    extract_table_schedule,
    extract_table_schedule_from_boxes,
    extract_natural_language_schedule,
    deduplicate_ocr_ops,
)
from app.routes.upload import extract_text_from_image_bytes

client = TestClient(app)

FONT_PATH = "/usr/share/fonts/ttf-readex-pro/ReadexPro-Regular.ttf"
FONT_SMALL = ImageFont.truetype(FONT_PATH, 24)
FONT_MEDIUM = ImageFont.truetype(FONT_PATH, 32)
FONT_LARGE = ImageFont.truetype(FONT_PATH, 40)


def create_schedule_image(text: str, fmt: str = "PNG", font=None, size=None) -> bytes:
    """Helper to generate an image buffer with rendered text."""
    if font is None:
        font = FONT_MEDIUM
    if size is None:
        size = (800, 160)
    img = Image.new("RGB", size, color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 40), text, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def create_table_image(headers, rows, fmt="PNG") -> bytes:
    """
    Create an image containing a transaction table.
    headers: list of strings e.g. ['T1', 'T2', 'T3']
    rows: list of dicts, each mapping header->operation or None
          e.g. [{'T1': 'R(A)', 'T2': None}, {'T1': None, 'T2': 'W(A)'}]
    """
    col_width = 180
    row_height = 55
    margin_x = 50
    margin_y = 30
    n_cols = len(headers)
    n_rows = len(rows)
    img_w = margin_x * 2 + col_width * n_cols
    img_h = margin_y * 2 + row_height * (n_rows + 1)

    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    font = FONT_MEDIUM

    # Draw headers
    for i, hdr in enumerate(headers):
        x = margin_x + i * col_width + col_width // 4
        d.text((x, margin_y), hdr, fill=(0, 0, 0), font=font)

    # Draw rows
    for row_idx, row in enumerate(rows):
        y = margin_y + (row_idx + 1) * row_height
        for col_idx, hdr in enumerate(headers):
            op = row.get(hdr)
            if op:
                x = margin_x + col_idx * col_width + col_width // 4
                d.text((x, y), op, fill=(0, 0, 0), font=font)

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ============================================================================
# 1. Inline schedule with transaction IDs
# ============================================================================
class TestInlineSchedules:
    def test_standard_inline_with_tx_ids(self):
        text = "R1(X), W2(X), R1(Y), W2(Y)"
        res = parse_schedule_from_text(text)
        assert res["parsed_schedule"] == "R1(X), W2(X), R1(Y), W2(Y)"
        assert res["operation_count"] == 4
        assert res["is_valid"] is True

    def test_space_separated_inline(self):
        text = "R1(X) W2(X) W1(X) W3(X)"
        res = parse_schedule_from_text(text)
        assert res["parsed_schedule"] == "R1(X), W2(X), W1(X), W3(X)"
        assert res["operation_count"] == 4
        assert res["is_valid"] is True

    def test_semicolon_separated(self):
        text = "R1(X); W2(X); W1(X); W3(X)"
        res = parse_schedule_from_text(text)
        assert res["parsed_schedule"] == "R1(X), W2(X), W1(X), W3(X)"
        assert res["operation_count"] == 4

    def test_newline_separated(self):
        text = "R1(X)\nW2(X)\nW1(X)\nW3(X)"
        res = parse_schedule_from_text(text)
        assert res["parsed_schedule"] == "R1(X), W2(X), W1(X), W3(X)"
        assert res["operation_count"] == 4

    def test_colon_notation(self):
        text = "T1: R(X), T2: W(X), T1: W(X), T3: W(X)"
        res = parse_schedule_from_text(text)
        assert res["parsed_schedule"] == "R1(X), W2(X), W1(X), W3(X)"
        assert res["is_valid"] is True

    def test_bracket_notation(self):
        text = "R1[X], W2[X], R1[Y], W2[Y]"
        res = parse_schedule_from_text(text)
        assert res["parsed_schedule"] == "R1(X), W2(X), R1(Y), W2(Y)"
        assert res["is_valid"] is True

    def test_lowercase_operations(self):
        text = "r1(x), w2(x), r1(y), w2(y)"
        res = parse_schedule_from_text(text)
        assert res["parsed_schedule"] == "R1(X), W2(X), R1(Y), W2(Y)"
        assert res["is_valid"] is True

    def test_labeled_schedule(self):
        text = "S1 = R1(X), W2(X), W1(X)"
        res = parse_schedule_from_text(text)
        assert "R1(X)" in res["parsed_schedule"]
        assert "W2(X)" in res["parsed_schedule"]
        assert "W1(X)" in res["parsed_schedule"]


# ============================================================================
# 2. Image OCR: clear inline schedule
# ============================================================================
class TestImageInlineOCR:
    def test_ocr_inline_no_duplication(self):
        """Verify OCR no longer duplicates text from multi-pass."""
        img_bytes = create_schedule_image("R1(X), W2(X), R1(Y), W2(Y)")
        text = extract_text_from_image_bytes(img_bytes)
        res = parse_schedule_from_text(text)
        assert res["operation_count"] == 4  # Was 8 with the old bug
        assert res["parsed_schedule"] == "R1(X), W2(X), R1(Y), W2(Y)"

    def test_ocr_with_question_context_no_duplication(self):
        """Inline schedule surrounded by question text should not duplicate."""
        img = Image.new("RGB", (900, 200), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((20, 20), "Q5. Consider the following schedule:", fill=(0, 0, 0), font=FONT_MEDIUM)
        d.text((20, 70), "S: R1(X), W2(X), W1(X), W3(X)", fill=(0, 0, 0), font=FONT_MEDIUM)
        d.text((20, 120), "Check if it is conflict serializable.", fill=(0, 0, 0), font=FONT_MEDIUM)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        text = extract_text_from_image_bytes(buf.getvalue())
        res = parse_schedule_from_text(text)
        assert res["operation_count"] == 4  # Was 8 with the old bug

    def test_upload_png_returns_correct_count(self):
        img_bytes = create_schedule_image("R1(X), W2(X), R3(Y), W1(Y)")
        response = client.post(
            "/api/upload/question",
            files={"file": ("question.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["operation_count"] == 4
        assert data["is_valid"] is True


# ============================================================================
# 3. Image OCR: T1/T2 transaction table
# ============================================================================
class TestImageTableOCR:
    def test_two_transaction_table_bounding_boxes(self):
        """Test bounding-box based table extraction for 2 transactions."""
        img_bytes = create_table_image(
            headers=["T1", "T2"],
            rows=[
                {"T1": "R(A)", "T2": None},
                {"T1": None, "T2": "R(A)"},
                {"T1": "W(A)", "T2": None},
                {"T1": None, "T2": "R(B)"},
                {"T1": None, "T2": "W(A)"},
                {"T1": "R(B)", "T2": None},
                {"T1": "W(B)", "T2": None},
                {"T1": None, "T2": "W(B)"},
            ]
        )
        ops = extract_table_schedule_from_boxes(img_bytes)
        assert len(ops) == 8
        # Verify T1 and T2 both appear
        t1_ops = [o for o in ops if o.startswith(("R1", "W1"))]
        t2_ops = [o for o in ops if o.startswith(("R2", "W2"))]
        assert len(t1_ops) >= 3
        assert len(t2_ops) >= 3

    def test_three_transaction_table_bounding_boxes(self):
        """Test bounding-box extraction for 3 transactions."""
        img_bytes = create_table_image(
            headers=["T1", "T2", "T3"],
            rows=[
                {"T1": "R(X)", "T2": None, "T3": None},
                {"T1": None, "T2": "W(X)", "T3": None},
                {"T1": None, "T2": None, "T3": "R(Y)"},
                {"T1": "W(X)", "T2": None, "T3": None},
            ]
        )
        ops = extract_table_schedule_from_boxes(img_bytes)
        assert len(ops) >= 3  # At minimum 3 recognized (OCR may miss some)
        # Verify multiple transactions are assigned
        tx_nums = set()
        for op in ops:
            m = re.match(r'[RW](\d+)', op)
            if m:
                tx_nums.add(m.group(1))
        assert len(tx_nums) >= 2


# ============================================================================
# 4. Text-based table parsing
# ============================================================================
class TestTextTableParsing:
    def test_text_table_two_transactions(self):
        text = "T1      T2\nR(A)\n        R(A)\nW(A)\n        R(B)"
        ops = extract_table_schedule(text)
        assert len(ops) == 4

    def test_text_table_three_transactions(self):
        text = "T1      T2      T3\nR(A)\n        W(A)\n                R(B)"
        ops = extract_table_schedule(text)
        assert len(ops) == 3


# ============================================================================
# 5. Natural language schedule extraction
# ============================================================================
class TestNaturalLanguage:
    def test_reads_writes_nl(self):
        text = "T1 reads X, T2 writes X, then T1 writes X"
        res = parse_schedule_from_text(text)
        assert res["operation_count"] == 3
        assert "R1(X)" in res["parsed_schedule"]
        assert "W2(X)" in res["parsed_schedule"]
        assert "W1(X)" in res["parsed_schedule"]

    def test_transaction_reads_writes(self):
        text = "Transaction 1 reads A, Transaction 2 writes A"
        ops = extract_natural_language_schedule(text)
        assert len(ops) == 2
        assert "R1(A)" in ops
        assert "W2(A)" in ops


# ============================================================================
# 6. Operations without explicit transaction IDs
# ============================================================================
class TestNoTransactionID:
    def test_bare_operations_flagged_as_uncertain(self):
        """R(A), W(A) without transaction IDs should be flagged as uncertain."""
        text = "R(A), W(A), R(B), W(B)"
        res = parse_schedule_from_text(text)
        assert len(res["uncertain_parts"]) > 0
        assert any("missing transaction" in u.lower() for u in res["uncertain_parts"])

    def test_incomplete_operations_detected(self):
        text = "R1(X), W(X), R1(Y)"
        res = parse_schedule_from_text(text)
        assert "R1(X)" in res["parsed_schedule"]
        assert "R1(Y)" in res["parsed_schedule"]
        assert any("missing transaction id" in u.lower() for u in res["uncertain_parts"])


# ============================================================================
# 7. Unrelated text should produce zero operations
# ============================================================================
class TestUnrelatedText:
    def test_no_operations_in_unrelated_text(self):
        text = "Welcome to the Database Management course midterm exam."
        res = parse_schedule_from_text(text)
        assert res["parsed_schedule"] == ""
        assert res["operation_count"] == 0
        assert res["is_valid"] is False

    def test_unrelated_image_ocr(self):
        img_bytes = create_schedule_image(
            "Database Systems chapter 5 notes without operations"
        )
        response = client.post(
            "/api/upload/question",
            files={"file": ("notes.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["parsed_schedule"] == ""
        assert data["operation_count"] == 0
        assert data["is_valid"] is False


# ============================================================================
# 8. Invalid or incomplete notation
# ============================================================================
class TestInvalidNotation:
    def test_empty_text(self):
        res = parse_schedule_from_text("")
        assert res["parsed_schedule"] == ""
        assert res["operation_count"] == 0
        assert res["is_valid"] is False

    def test_incomplete_operation_no_item(self):
        """R1 or W2 without parentheses/data item should be flagged."""
        text = "R1, W2, R3"
        res = parse_schedule_from_text(text)
        assert res["operation_count"] == 0
        assert len(res["uncertain_parts"]) > 0


# ============================================================================
# 9. OCR text exists but zero operations recognized
# ============================================================================
class TestOCRSuccessParseFailure:
    def test_ocr_produces_text_no_schedule(self):
        """When OCR extracts text but no operations are found, the message must
        distinguish this from total OCR failure."""
        img_bytes = create_schedule_image("Explain the concept of two-phase locking.")
        response = client.post(
            "/api/upload/question",
            files={"file": ("notes.png", img_bytes, "image/png")}
        )
        data = response.json()
        assert data["success"] is True
        assert data["operation_count"] == 0
        assert data["ocr_succeeded"] is True
        assert "text was recognized" in data["message"].lower() or \
               "no valid" in data["message"].lower() or \
               "text was detected" in data["message"].lower()


# ============================================================================
# 10. OCR deduplication
# ============================================================================
class TestOCRDeduplication:
    def test_duplicate_text_removal(self):
        text = "R1(X), W2(X)\n\nR1(X), W2(X)"
        deduped = deduplicate_ocr_ops(text)
        assert deduped.count("R1(X)") == 1

    def test_non_duplicate_preserved(self):
        text = "R1(X), W2(X)\nR3(Y), W1(Y)"
        deduped = deduplicate_ocr_ops(text)
        assert "R1(X)" in deduped
        assert "R3(Y)" in deduped


# ============================================================================
# 11. Multiple schedules in one document
# ============================================================================
class TestMultipleSchedules:
    def test_multiple_labeled_schedules(self):
        text = "S1: R1(X), W2(X), W1(X)\nS2: R3(Y), W1(Y), R2(Y)"
        res = parse_schedule_from_text(text)
        assert len(res["candidate_schedules"]) >= 2

    def test_labeled_plus_inline_candidates(self):
        text = "Schedule A: R1(X), W2(X)\nAlso consider R3(Y), W1(Y)"
        res = parse_schedule_from_text(text)
        assert len(res["candidate_schedules"]) >= 1


# ============================================================================
# 12. Full upload-to-analysis pipeline
# ============================================================================
class TestEndToEndPipeline:
    def test_upload_and_analyze_standard_schedule(self):
        """Test that upload → extract → confirm → analyze works end to end."""
        img_bytes = create_schedule_image("R1(X), W2(X), W1(X), W3(X)")
        # Step 1: Upload and extract
        upload_resp = client.post(
            "/api/upload/question",
            files={"file": ("q.png", img_bytes, "image/png")}
        )
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        assert upload_data["operation_count"] >= 3
        assert upload_data["is_valid"] is True

        schedule_text = upload_data["parsed_schedule"]

        # Step 2: Analyze the extracted schedule using the main analyzer
        analyze_resp = client.post(
            "/api/analyze",
            json={"schedule_text": schedule_text}
        )
        assert analyze_resp.status_code == 200
        analysis = analyze_resp.json()
        assert analysis["success"] is True
        assert analysis["operation_count"] >= 3
        assert "conflict_serializable" in analysis
        assert "view_serializable" in analysis

    def test_upload_invalid_extension(self):
        response = client.post(
            "/api/upload/question",
            files={"file": ("test.docx", b"R1(X), W2(X)", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]


    def test_upload_empty_file(self):
        response = client.post(
            "/api/upload/question",
            files={"file": ("test.png", b"", "image/png")}
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"]

    def test_upload_webp_format(self):
        img_bytes = create_schedule_image("R1(X), W2(X), W1(X), W3(X)", fmt="WEBP")
        response = client.post(
            "/api/upload/question",
            files={"file": ("question.webp", img_bytes, "image/webp")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "R1(X)" in data["parsed_schedule"]
        assert data["is_valid"] is True

    def test_upload_jpg_format(self):
        img_bytes = create_schedule_image("R1(X), W2(X), R1(Y), W2(Y)", fmt="JPEG")
        response = client.post(
            "/api/upload/question",
            files={"file": ("question.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "R1(X)" in data["parsed_schedule"]


# ============================================================================
# 13. PDF upload
# ============================================================================
class TestPDFUpload:
    def test_pdf_digital_text(self):
        from reportlab.pdfgen import canvas as pdf_canvas
        buf = io.BytesIO()
        c = pdf_canvas.Canvas(buf)
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


# ============================================================================
# 14. Plain text file (.txt) upload
# ============================================================================
class TestTextFileUpload:
    def test_txt_file_inline_schedule(self):
        txt_content = b"Schedule S: R1(A), W2(A), R1(B), W2(B)"
        response = client.post(
            "/api/upload/question",
            files={"file": ("schedule.txt", txt_content, "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["file_type"] == "text"
        assert data["operation_count"] == 4
        assert data["parsed_schedule"] == "R1(A), W2(A), R1(B), W2(B)"
        assert data["is_valid"] is True

    def test_txt_file_tabular_schedule(self):
        txt_content = b"T1\tT2\nR(A)\n\tR(A)\nW(A)\n\tW(B)"
        response = client.post(
            "/api/upload/question",
            files={"file": ("table_schedule.txt", txt_content, "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["file_type"] == "text"
        assert data["operation_count"] == 4
        assert "R1(A)" in data["parsed_schedule"]
        assert "R2(A)" in data["parsed_schedule"]
        assert data["is_valid"] is True


# ============================================================================
# 14. Operations with line breaks between them
# ============================================================================
class TestLineBreakSeparated:
    def test_operations_separated_by_line_breaks(self):
        text = "R1(X)\nW2(X)\nW1(X)\nW3(X)"
        res = parse_schedule_from_text(text)
        assert res["operation_count"] == 4
        assert res["is_valid"] is True

    def test_operations_with_mixed_separators(self):
        text = "R1(X), W2(X)\nW1(X); W3(X)"
        res = parse_schedule_from_text(text)
        assert res["operation_count"] == 4


# ============================================================================
# 15. Primary Screenshot Regression Tests (Ground Truth Verification)
# ============================================================================
class TestPrimaryRegressionScreenshot:
    """
    Primary regression test suite using the exact screenshot from Phase 1.
    Ground Truth Schedule:
      R1(X), W2(X), R3(Y), W1(X), W2(Y), R1(Y), W3(X)
    Must contain exactly 7 operations in exact order.
    No operations missing (W1(X), W2(Y) must be present).
    No operations duplicated (R3(Y) must appear exactly once).
    """

    FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "primary_regression_screenshot.png")
    EXPECTED_SCHEDULE = "R1(X), W2(X), R3(Y), W1(X), W2(Y), R1(Y), W3(X)"
    EXPECTED_OPS = ["R1(X)", "W2(X)", "R3(Y)", "W1(X)", "W2(Y)", "R1(Y)", "W3(X)"]

    def test_primary_screenshot_extracts_exact_seven_operations(self):
        assert os.path.exists(self.FIXTURE_PATH), f"Fixture not found at {self.FIXTURE_PATH}"

        with open(self.FIXTURE_PATH, "rb") as f:
            response = client.post(
                "/api/upload/question",
                files={"file": ("primary_regression_screenshot.png", f, "image/png")}
            )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["file_type"] == "image"
        assert data["ocr_succeeded"] is True

        extracted_schedule = data["parsed_schedule"]
        assert extracted_schedule == self.EXPECTED_SCHEDULE, (
            f"Expected '{self.EXPECTED_SCHEDULE}', but got '{extracted_schedule}'"
        )

        extracted_ops = [op.strip() for op in extracted_schedule.split(",")]
        assert len(extracted_ops) == 7, f"Expected exactly 7 operations, got {len(extracted_ops)}"
        assert data["operation_count"] == 7

        # Verify operations in exact sequential order
        for idx, (expected_op, actual_op) in enumerate(zip(self.EXPECTED_OPS, extracted_ops)):
            assert actual_op == expected_op, (
                f"Operation at index {idx} mismatch: expected {expected_op}, got {actual_op}"
            )

        # Explicit negative assertions from the bug report
        assert "W1(X)" in extracted_ops, "W1(X) was missing in old version but must be present"
        assert "W2(Y)" in extracted_ops, "W2(Y) was missing in old version but must be present"
        assert extracted_ops.count("R3(Y)") == 1, (
            f"R3(Y) must appear exactly once, but appeared {extracted_ops.count('R3(Y)')} times"
        )

        # High confidence check
        assert data["confidence_score"] >= 0.85
        assert data["status"] in ("exact_match", "high_confidence")

    def test_primary_screenshot_end_to_end_analysis(self):
        """Verify the extracted schedule feeds directly into serializability analysis."""
        with open(self.FIXTURE_PATH, "rb") as f:
            upload_resp = client.post(
                "/api/upload/question",
                files={"file": ("primary_regression_screenshot.png", f, "image/png")}
            )
        assert upload_resp.status_code == 200
        schedule_text = upload_resp.json()["parsed_schedule"]

        # Run through existing analyzer
        analyze_resp = client.post(
            "/api/analyze",
            json={"schedule_text": schedule_text}
        )
        assert analyze_resp.status_code == 200
        analysis = analyze_resp.json()

        # The mathematical serializability behavior must be deterministic
        assert "conflict_serializable" in analysis
        assert "view_serializable" in analysis
        assert "precedence_graph" in analysis

        nodes = analysis["precedence_graph"]["nodes"]
        node_ids = {n["data"]["id"] for n in nodes}
        assert {"T1", "T2", "T3"}.issubset(node_ids)


# ============================================================================
# 11. Real Regression Fixture: Two-Transaction Table Worksheet
# ============================================================================
class TestTwoTransactionTableWorksheet:
    """
    Regression test suite for the two-transaction table worksheet screenshot.
    Fixture image: backend/tests/fixtures/table_schedule_two_tx.png
    Expected operations: R1(A), R2(A), W2(A), W1(A), R1(B), R2(B), W1(B), W2(B)
    Expected count: exactly 8 operations in row-by-row chronological order.
    """

    FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "table_schedule_two_tx.png")
    EXPECTED_SCHEDULE = "R1(A), R2(A), W2(A), W1(A), R1(B), R2(B), W1(B), W2(B)"
    EXPECTED_OPS = ["R1(A)", "R2(A)", "W2(A)", "W1(A)", "R1(B)", "R2(B)", "W1(B)", "W2(B)"]

    def test_two_tx_table_extracts_exact_eight_operations(self):
        assert os.path.exists(self.FIXTURE_PATH), f"Fixture not found at {self.FIXTURE_PATH}"

        with open(self.FIXTURE_PATH, "rb") as f:
            response = client.post(
                "/api/upload/question",
                files={"file": ("table_schedule_two_tx.png", f, "image/png")}
            )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["file_type"] == "image"
        assert data["ocr_succeeded"] is True

        extracted_schedule = data["parsed_schedule"]
        assert extracted_schedule == self.EXPECTED_SCHEDULE, (
            f"Expected '{self.EXPECTED_SCHEDULE}', but got '{extracted_schedule}'"
        )

        extracted_ops = [op.strip() for op in extracted_schedule.split(",")]
        assert len(extracted_ops) == 8, f"Expected exactly 8 operations, got {len(extracted_ops)}"
        assert data["operation_count"] == 8

        # Verify operations in exact sequential order
        for idx, (expected_op, actual_op) in enumerate(zip(self.EXPECTED_OPS, extracted_ops)):
            assert actual_op == expected_op, (
                f"Operation at index {idx} mismatch: expected {expected_op}, got {actual_op}"
            )

        # Ensure confidence and status
        assert data["confidence_score"] >= 0.85
        assert data["status"] in ("exact_match", "high_confidence")

    def test_two_tx_table_end_to_end_analysis(self):
        """Verify the 8-operation schedule feeds directly into serializability analysis."""
        with open(self.FIXTURE_PATH, "rb") as f:
            upload_resp = client.post(
                "/api/upload/question",
                files={"file": ("table_schedule_two_tx.png", f, "image/png")}
            )
        assert upload_resp.status_code == 200
        schedule_text = upload_resp.json()["parsed_schedule"]

        # Run through existing analyzer
        analyze_resp = client.post(
            "/api/analyze",
            json={"schedule_text": schedule_text}
        )
        assert analyze_resp.status_code == 200
        analysis = analyze_resp.json()

        assert "conflict_serializable" in analysis
        assert "view_serializable" in analysis
        assert "precedence_graph" in analysis

        nodes = analysis["precedence_graph"]["nodes"]
        node_ids = {n["data"]["id"] for n in nodes}
        assert {"T1", "T2"}.issubset(node_ids)


# ============================================================================
# 12. Honest Accuracy Evaluation Metrics
# ============================================================================
class TestHonestAccuracyEvaluation:
    def test_evaluation_metrics_on_test_dataset(self):
        """
        Evaluation metrics on ground-truth fixtures:
        Calculates match rate, precision, recall, missing ops, extra ops, and order accuracy.
        """
        dataset = [
            {
                "name": "Primary Slide Screenshot",
                "file_path": os.path.join(os.path.dirname(__file__), "fixtures", "primary_regression_screenshot.png"),
                "expected": ["R1(X)", "W2(X)", "R3(Y)", "W1(X)", "W2(Y)", "R1(Y)", "W3(X)"],
            },
            {
                "name": "Two-Transaction Table Worksheet",
                "file_path": os.path.join(os.path.dirname(__file__), "fixtures", "table_schedule_two_tx.png"),
                "expected": ["R1(A)", "R2(A)", "W2(A)", "W1(A)", "R1(B)", "R2(B)", "W1(B)", "W2(B)"],
            }
        ]

        for item in dataset:
            assert os.path.exists(item["file_path"]), f"Fixture missing: {item['file_path']}"
            with open(item["file_path"], "rb") as f:
                res = client.post(
                    "/api/upload/question",
                    files={"file": (os.path.basename(item["file_path"]), f, "image/png")}
                )
            assert res.status_code == 200
            data = res.json()
            actual_ops = [op.strip() for op in data["parsed_schedule"].split(",") if op.strip()]
            expected_ops = item["expected"]

            # Compute precision, recall, missing, extra
            matches = sum(1 for a, e in zip(actual_ops, expected_ops) if a == e)
            missing = max(0, len(expected_ops) - len(actual_ops))
            extra = max(0, len(actual_ops) - len(expected_ops))
            order_accuracy = (matches / len(expected_ops)) if expected_ops else 1.0

            assert matches == len(expected_ops), f"Mismatch in {item['name']}: {actual_ops} vs {expected_ops}"
            assert missing == 0
            assert extra == 0
            assert order_accuracy == 1.0

