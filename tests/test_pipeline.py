"""
Comprehensive Test Suite for ParseX Pipeline.
Validates all 7 test cases:
1. Normal digital PDF
2. Scanned PDF (image-only, no text layer)
3. Handwritten page (Page 10 AVL tree rotation)
4. Page containing table
5. Page containing chart/diagram
6. Mixed digital + visual page
7. Corrupt/unsupported input
"""

import os
import unittest
from parsex.router import DocumentPipeline
from parsex.schema import document_to_markdown
from parsex.qa import answer_question

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")


class TestParseXPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = DocumentPipeline()

    def test_01_digital_pdf(self):
        """Test 1: Normal Digital PDF extraction via digital parser."""
        pdf_path = os.path.join(SAMPLES_DIR, "test1_digital.pdf")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = self.pipeline.process_document(pdf_bytes, filename="test1_digital.pdf")
        self.assertEqual(doc["pages"], 1)
        self.assertTrue(len(doc["blocks"]) > 0)

        # Check that digital text parser was utilized
        has_digital = any(b["extractor"] in ("digital-text", "pdf-text") for b in doc["blocks"])
        self.assertTrue(has_digital, "Expected digital text extractor to be used on digital PDF.")

        # Check provenance
        for b in doc["blocks"]:
            self.assertIsNotNone(b["bbox"])
            self.assertEqual(len(b["bbox"]), 4)
            self.assertGreaterEqual(b["confidence"], 0.80)

        # Check Markdown generation
        md = document_to_markdown(doc)
        self.assertIn("Enterprise System Architecture", md)
        print("✓ Test 1: Digital PDF passed.")

    def test_02_scanned_pdf(self):
        """Test 2: Scanned PDF (image-only) routed to high-res rendering and OCR."""
        pdf_path = os.path.join(SAMPLES_DIR, "test2_scanned.pdf")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = self.pipeline.process_document(pdf_bytes, filename="test2_scanned.pdf")
        self.assertEqual(doc["pages"], 1)
        self.assertTrue(len(doc["blocks"]) > 0)

        # Verify OCR extractor was invoked
        has_ocr = any("ocr" in b["extractor"] for b in doc["blocks"])
        self.assertTrue(has_ocr, "Expected OCR engine to extract text from scanned PDF.")

        # Verify OCR preserved text & bboxes
        ocr_blocks = [b for b in doc["blocks"] if "ocr" in b["extractor"]]
        full_text = " ".join(str(b["content"]) for b in ocr_blocks).upper()
        self.assertTrue(
            "AUDIT" in full_text or "COMPLIANCE" in full_text or "REPORT" in full_text or "MEMORANDUM" in full_text,
            f"Expected OCR to read text from scanned page, got: {full_text}",
        )
        print("✓ Test 2: Scanned PDF passed.")

    def test_03_handwritten_page10_avl(self):
        """Test 3: Handwritten Page 10 AVL Tree Rotation page."""
        pdf_path = os.path.join(SAMPLES_DIR, "test3_handwritten_avl.pdf")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = self.pipeline.process_document(pdf_bytes, filename="test3_handwritten_avl.pdf")
        self.assertEqual(doc["pages"], 10)

        # Page 10 must NOT simply be classified as an unavailable figure!
        page10_blocks = [b for b in doc["blocks"] if b["page"] == 10]
        self.assertTrue(len(page10_blocks) >= 2, f"Expected multiple blocks on Page 10, got {len(page10_blocks)}")

        block_types = [b["type"] for b in page10_blocks]
        print(f"Page 10 detected block types: {block_types}")

        # Check that handwriting or diagram was extracted
        has_diagram = any(b["type"] == "diagram" for b in page10_blocks)
        has_handwriting = any(b["type"] == "handwritten_text" for b in page10_blocks)
        has_equation = any(b["type"] == "equation" for b in page10_blocks)

        self.assertTrue(has_diagram or has_handwriting, "Expected diagram or handwriting on Page 10.")

        # Check diagram content and structure
        diagram_block = next((b for b in page10_blocks if b["type"] == "diagram"), None)
        if diagram_block:
            c = diagram_block["content"]
            self.assertIn("AVL", c.get("title", "") + c.get("description", ""))
            self.assertTrue(len(c.get("elements", [])) > 0)
            self.assertTrue(len(c.get("relationships", [])) > 0)

        # Test Q&A on handwritten / diagram content
        q_res = answer_question(doc, "What rotation is shown in the diagram on page 10?")
        self.assertIn("rotation", q_res["answer"].lower())
        self.assertIn("right", q_res["answer"].lower())
        self.assertEqual(q_res["page"], 10)
        self.assertIsNotNone(q_res["bbox"])

        # Check Markdown formatting
        md = document_to_markdown(doc)
        self.assertIn("Page 10", md)
        print("✓ Test 3: Handwritten Page 10 AVL page passed.")

    def test_04_table_document(self):
        """Test 4: Table Document extraction with row and column alignment."""
        pdf_path = os.path.join(SAMPLES_DIR, "test4_table.pdf")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = self.pipeline.process_document(pdf_bytes, filename="test4_table.pdf")
        self.assertTrue(len(doc["blocks"]) > 0)

        # Check that table block exists
        table_blocks = [b for b in doc["blocks"] if b["type"] == "table"]
        self.assertTrue(len(table_blocks) > 0, "Expected table block in table document.")

        tbl = table_blocks[0]
        self.assertIsInstance(tbl["content"], list)
        self.assertGreaterEqual(len(tbl["content"]), 2)

        # Check Q&A against table cell
        q_res = answer_question(doc, "What was the Total debt in FY24?")
        self.assertIsNotNone(q_res["answer"])
        print("✓ Test 4: Table document passed.")

    def test_05_diagram_chart_document(self):
        """Test 5: Chart & Diagram Document."""
        pdf_path = os.path.join(SAMPLES_DIR, "test5_diagram_chart.pdf")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = self.pipeline.process_document(pdf_bytes, filename="test5_diagram_chart.pdf")
        visual_blocks = [b for b in doc["blocks"] if b["type"] in ("chart", "diagram", "figure")]
        self.assertTrue(len(visual_blocks) > 0, "Expected chart or visual block.")
        print("✓ Test 5: Chart and diagram document passed.")

    def test_06_mixed_page(self):
        """Test 6: Mixed Digital + Visual + Handwritten Page."""
        pdf_path = os.path.join(SAMPLES_DIR, "test6_mixed_page.pdf")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = self.pipeline.process_document(pdf_bytes, filename="test6_mixed_page.pdf")
        # Mixed page must contain both text and visual/handwritten regions
        types = set(b["type"] for b in doc["blocks"])
        self.assertTrue(len(types) >= 2, f"Expected multiple region types on mixed page, got: {types}")
        print("✓ Test 6: Mixed page passed.")

    def test_07_corrupted_input(self):
        """Test 7: Corrupted or invalid input handling without crash."""
        corrupt_bytes = b"CORRUPTED_NON_PDF_DATA"
        with self.assertRaises(Exception):
            self.pipeline.process_document(corrupt_bytes, filename="corrupt.pdf")
        print("✓ Test 7: Corrupted input gracefully caught.")


if __name__ == "__main__":
    unittest.main()
