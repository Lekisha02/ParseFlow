"""
PyMuPDF PDF High-Resolution Rendering and Text-Layer Inspection.
Renders pages to 200-300 DPI high-resolution imagery when text layer is absent
or insufficient, preserving exact page geometry and provenance.
"""

import io
import base64
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import pymupdf


# Standard PDF DPI is 72. 216 DPI = 3.0x scale, 300 DPI = ~4.17x scale.
DEFAULT_RENDER_DPI = 216
POINTS_PER_INCH = 72.0


class PageRenderResult:
    def __init__(
        self,
        page_num: int,
        image: Image.Image,
        page_width: float,
        page_height: float,
        image_width: int,
        image_height: int,
        scale: float,
        dpi: int,
        document_id: str,
    ):
        self.page_num = page_num
        self.image = image
        self.page_width = page_width
        self.page_height = page_height
        self.image_width = image_width
        self.image_height = image_height
        self.scale = scale
        self.dpi = dpi
        self.document_id = document_id

    def crop_region_pdf_coords(self, bbox: List[float]) -> Image.Image:
        """Crop an image region specified in PDF point coordinates [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        # Convert points to pixel coordinates
        px1 = max(0, int(x1 * self.scale))
        py1 = max(0, int(y1 * self.scale))
        px2 = min(self.image_width, int(x2 * self.scale))
        py2 = min(self.image_height, int(y2 * self.scale))

        # Add small margin for better handwriting/diagram context
        margin = int(4 * self.scale)
        px1 = max(0, px1 - margin)
        py1 = max(0, py1 - margin)
        px2 = min(self.image_width, px2 + margin)
        py2 = min(self.image_height, py2 + margin)

        if px2 <= px1 or py2 <= py1:
            return self.image

        return self.image.crop((px1, py1, px2, py2))

    def crop_to_base64_jpeg(self, bbox: List[float], quality: int = 85) -> str:
        """Return base64-encoded JPEG crop for vision model ingestion or display."""
        cropped = self.crop_region_pdf_coords(bbox)
        buf = io.BytesIO()
        cropped.convert("RGB").save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


class PDFRenderer:
    def __init__(self, target_dpi: int = DEFAULT_RENDER_DPI):
        self.target_dpi = target_dpi
        self.scale = target_dpi / POINTS_PER_INCH

    def inspect_text_layer(self, page: pymupdf.Page) -> Dict[str, Any]:
        """
        Determine if the page contains a usable digital text layer or is
        a scanned / image-only / handwritten page requiring visual OCR pipeline.
        """
        words = page.get_text("words")
        text = page.get_text("text").strip()
        word_count = len(words)
        char_count = len(text)

        rect = page.rect
        page_area = max(1.0, rect.width * rect.height)

        # Calculate bounding box union of all text to measure coverage
        text_area = 0.0
        if words:
            for w in words:
                w_area = max(0.0, (w[2] - w[0]) * (w[3] - w[1]))
                text_area += w_area

        coverage = text_area / page_area

        # Page has usable text layer if it contains legitimate digital words (>= 8)
        # Scanned pages and pure image drawings have 0 (or near 0) digital words.
        has_usable_text = word_count >= 8 and char_count >= 20

        return {
            "has_usable_text": has_usable_text,
            "word_count": word_count,
            "char_count": char_count,
            "coverage": coverage,
            "raw_text": text,
            "words": words,
        }

    def render_page(
        self,
        page: pymupdf.Page,
        page_num: int,
        document_id: str = "doc_001",
        dpi: Optional[int] = None,
    ) -> PageRenderResult:
        """
        Render a PDF page to a high-resolution PIL Image (~200-300 DPI).
        Preserves page number, image dimensions, scale, and document ID.
        """
        used_dpi = dpi or self.target_dpi
        scale = used_dpi / POINTS_PER_INCH
        mat = pymupdf.Matrix(scale, scale)

        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        return PageRenderResult(
            page_num=page_num,
            image=img,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            image_width=pix.width,
            image_height=pix.height,
            scale=scale,
            dpi=used_dpi,
            document_id=document_id,
        )

    def image_to_base64_jpeg(self, img: Image.Image, quality: int = 85) -> str:
        """Convert a PIL Image to base64-encoded JPEG."""
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
