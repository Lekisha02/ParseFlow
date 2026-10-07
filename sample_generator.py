"""
Sample Document Generator for ParseX Test Suite.
Generates comprehensive synthetic test documents:
- Test 1: Clean Digital PDF
- Test 2: Scanned Printed PDF (no text layer)
- Test 3: Handwritten AVL Page (Page 10 demo test case)
- Test 4: Financial Table PDF
- Test 5: Diagram & Chart PDF
- Test 6: Mixed Digital + Visual + Handwritten PDF
- Test 7: Corrupted / invalid document
"""

import os
import io
import pymupdf
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def ensure_samples_dir():
    os.makedirs(SAMPLES_DIR, exist_ok=True)


def create_avl_page_image() -> Image.Image:
    """Create the synthetic handwritten AVL tree image matching the ParseX demo."""
    img = Image.new("RGB", (1000, 1300), color=(251, 250, 244))
    draw = ImageDraw.Draw(img)
    color = (28, 42, 107)

    # 1. Heading
    draw.text((60, 80), "AVL Rotations", fill=color)

    # 2. Handwritten C Code
    code_lines = [
        "struct Node* rotRight(struct Node* y){",
        "   struct Node* x = y->left;",
        "   y->left = x->right;",
        "   x->right = y;",
        "   return x; }",
    ]
    for i, line in enumerate(code_lines):
        draw.text((70, 170 + i * 50), line, fill=color)

    # 3. Label: LL Case
    draw.text((70, 500), "LL case (left-left heavy):", fill=color)

    # 4. Draw Left Tree: 30 -> 20 -> 10
    # Branches
    draw.line([(300, 650), (240, 750)], fill=color, width=3)
    draw.line([(240, 750), (180, 850)], fill=color, width=3)

    # Nodes
    for val, x, y in [("30", 300, 650), ("20", 240, 750), ("10", 180, 850)]:
        draw.ellipse([x - 32, y - 32, x + 32, y + 32], outline=color, fill=(251, 250, 244), width=3)
        draw.text((x - 12, y - 10), val, fill=color)

    # 5. Directed Transition Arrow: 'right rotate'
    draw.line([(420, 750), (580, 750)], fill=color, width=3)
    draw.polygon([(580, 750), (560, 735), (560, 765)], fill=color)
    draw.text((440, 720), "right rotate", fill=color)

    # 6. Draw Right Tree (Balanced): 20 -> 10, 30
    draw.line([(780, 650), (700, 750)], fill=color, width=3)
    draw.line([(780, 650), (860, 750)], fill=color, width=3)

    for val, x, y in [("20", 780, 650), ("10", 700, 750), ("30", 860, 750)]:
        draw.ellipse([x - 32, y - 32, x + 32, y + 32], outline=color, fill=(251, 250, 244), width=3)
        draw.text((x - 12, y - 10), val, fill=color)

    # 7. Mathematical & algorithmic balance factor notes
    draw.text((70, 980), "bf(30) = h(L) - h(R) = 2  -> unbalanced", fill=color)
    draw.text((70, 1050), "after rotation all bf = 0, balanced", fill=color)

    return img


def generate_all_samples():
    ensure_samples_dir()

    # -------------------------------------------------------------
    # Test 1: Normal Digital PDF
    # -------------------------------------------------------------
    doc1 = pymupdf.open()
    page1 = doc1.new_page(width=612, height=792)
    text1 = (
        "Enterprise System Architecture FY24\n\n"
        "Executive Summary\n"
        "The distributed platform scales horizontally across 12 availability zones. "
        "System reliability reached 99.995% during peak quarters.\n\n"
        "Core Capabilities:\n"
        "• Real-time consensus and replication\n"
        "• Cryptographic evidence provenance\n"
        "• Multi-tenant isolated storage\n"
    )
    page1.insert_text((50, 70), text1, fontsize=12)
    test1_path = os.path.join(SAMPLES_DIR, "test1_digital.pdf")
    doc1.save(test1_path)
    doc1.close()

    # -------------------------------------------------------------
    # Test 2: Scanned Printed PDF (Image Only, No Text Layer)
    # -------------------------------------------------------------
    scanned_img = Image.new("RGB", (800, 1000), color=(245, 244, 240))
    d2 = ImageDraw.Draw(scanned_img)
    d2.text((60, 80), "OFFICIAL AUDIT MEMORANDUM - CONFIDENTIAL", fill=(30, 30, 30))
    d2.text(
        (60, 140),
        "This scanned report certifies compliance with regulatory requirements.\n"
        "All transactions were inspected against primary ledgers.\n"
        "Internal controls were deemed effective with zero exceptions noted.",
        fill=(40, 40, 40),
    )
    d2.text((60, 300), "Auditor Signature: John Doe, CPA", fill=(40, 40, 40))

    doc2 = pymupdf.open()
    buf2 = io.BytesIO()
    scanned_img.save(buf2, format="JPEG", quality=85)
    img_doc2 = pymupdf.open(stream=buf2.getvalue(), filetype="jpeg")
    rect2 = pymupdf.Rect(0, 0, 612, 792)
    p2 = doc2.new_page(width=612, height=792)
    p2.insert_image(rect2, stream=buf2.getvalue())
    test2_path = os.path.join(SAMPLES_DIR, "test2_scanned.pdf")
    doc2.save(test2_path)
    doc2.close()

    # -------------------------------------------------------------
    # Test 3: Handwritten Page 10 AVL Tree Document
    # A 10-page document where Page 10 is the handwritten AVL rotation!
    # -------------------------------------------------------------
    doc3 = pymupdf.open()
    for i in range(1, 10):
        p = doc3.new_page(width=612, height=792)
        p.insert_text(
            (50, 70),
            f"Chapter 4: Advanced Binary Search Trees (Page {i})\n\n"
            f"Section {i}.1 covers tree balancing principles and rotation mechanics.",
            fontsize=12,
        )

    # Page 10: The Handwritten AVL Page!
    avl_img = create_avl_page_image()
    buf3 = io.BytesIO()
    avl_img.save(buf3, format="PNG")
    p10 = doc3.new_page(width=612, height=792)
    p10.insert_image(pymupdf.Rect(0, 0, 612, 792), stream=buf3.getvalue())

    test3_path = os.path.join(SAMPLES_DIR, "test3_handwritten_avl.pdf")
    doc3.save(test3_path)
    doc3.close()

    # Also save standalone PNG of the AVL page for quick direct image testing
    avl_img.save(os.path.join(SAMPLES_DIR, "test3_handwritten_page10.png"))

    # -------------------------------------------------------------
    # Test 4: Financial Table Document
    # -------------------------------------------------------------
    doc4 = pymupdf.open()
    p4 = doc4.new_page(width=612, height=792)
    table_text = (
        "Consolidated Balance Sheet FY24\n\n"
        "Financial Table (amounts in ₹ Cr):\n"
    )
    p4.insert_text((50, 60), table_text, fontsize=14)

    # Draw tabular lines and cell text
    table_rows = [
        ["Instrument", "FY23 (₹ Cr)", "FY24 (₹ Cr)"],
        ["Term loan", "3.0", "2.8"],
        ["Bonds", "2.5", "2.5"],
        ["Working capital", "1.2", "0.9"],
        ["Total debt", "6.7", "6.2"],
    ]
    y_start = 120
    row_h = 24
    col_w = 140
    for r_idx, row in enumerate(table_rows):
        y = y_start + r_idx * row_h
        for c_idx, val in enumerate(row):
            x = 50 + c_idx * col_w
            p4.insert_text((x + 5, y + 16), val, fontsize=10)
        p4.draw_line(pymupdf.Point(50, y + row_h), pymupdf.Point(50 + 3 * col_w, y + row_h))

    test4_path = os.path.join(SAMPLES_DIR, "test4_table.pdf")
    doc4.save(test4_path)
    doc4.close()

    # -------------------------------------------------------------
    # Test 5: Diagram & Chart PDF
    # -------------------------------------------------------------
    chart_img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    d5 = ImageDraw.Draw(chart_img)
    d5.text((50, 40), "Revenue by Region FY23 vs FY24", fill=(20, 20, 20))
    # Bars
    regions = [("North", 4.2, 5.1), ("South", 3.9, 4.8), ("East", 3.1, 3.5), ("West", 3.9, 5.0)]
    for i, (name, v1, v2) in enumerate(regions):
        x = 80 + i * 170
        d5.rectangle([x, 400 - int(v1 * 50), x + 40, 400], fill=(100, 140, 220))
        d5.rectangle([x + 45, 400 - int(v2 * 50), x + 85, 400], fill=(40, 80, 180))
        d5.text((x + 10, 415), name, fill=(30, 30, 30))
        d5.text((x + 5, 380 - int(v1 * 50)), str(v1), fill=(30, 30, 30))
        d5.text((x + 50, 380 - int(v2 * 50)), str(v2), fill=(30, 30, 30))

    buf5 = io.BytesIO()
    chart_img.save(buf5, format="PNG")
    doc5 = pymupdf.open()
    p5 = doc5.new_page(width=612, height=792)
    p5.insert_text((50, 50), "Performance Analytics", fontsize=16)
    p5.insert_image(pymupdf.Rect(50, 90, 562, 450), stream=buf5.getvalue())
    test5_path = os.path.join(SAMPLES_DIR, "test5_diagram_chart.pdf")
    doc5.save(test5_path)
    doc5.close()

    # -------------------------------------------------------------
    # Test 6: Mixed Digital + Visual + Handwritten Page
    # -------------------------------------------------------------
    doc6 = pymupdf.open()
    p6 = doc6.new_page(width=612, height=792)
    p6.insert_text((50, 60), "Algorithm Analysis: Tree Balances", fontsize=16)
    p6.insert_text(
        (50, 90),
        "This section documents self-balancing binary search trees. "
        "Notice the handwritten algorithm and rotation diagram below:",
        fontsize=11,
    )
    # Insert AVL crop in lower half
    p6.insert_image(pymupdf.Rect(50, 130, 562, 720), stream=buf3.getvalue())
    test6_path = os.path.join(SAMPLES_DIR, "test6_mixed_page.pdf")
    doc6.save(test6_path)
    doc6.close()

    # -------------------------------------------------------------
    # Test 7: Corrupted / Invalid File
    # -------------------------------------------------------------
    test7_path = os.path.join(SAMPLES_DIR, "test7_corrupt.pdf")
    with open(test7_path, "wb") as f:
        f.write(b"NOT_A_VALID_PDF_HEADER_CORRUPTED_STREAM_1234567890")

    print("All sample files generated successfully in:", SAMPLES_DIR)


if __name__ == "__main__":
    generate_all_samples()
