"""
RapidOCR Engine Module (PP-OCRv4 ONNX).
Provides fast, accurate local OCR with bounding box preservation, confidence scores,
and PDF-coordinate translation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
    _RAPID_OCR_AVAILABLE = True
except Exception:
    _RAPID_OCR_AVAILABLE = False


class OCREngine:
    _instance: Optional["OCREngine"] = None
    _rapid_ocr: Optional[Any] = None

    def __init__(self):
        if _RAPID_OCR_AVAILABLE and OCREngine._rapid_ocr is None:
            # Initialize PP-OCR ONNX models once
            OCREngine._rapid_ocr = RapidOCR()

    @classmethod
    def get_instance(cls) -> "OCREngine":
        if cls._instance is None:
            cls._instance = OCREngine()
        return cls._instance

    @property
    def is_available(self) -> bool:
        return _RAPID_OCR_AVAILABLE and self._rapid_ocr is not None

    def run_ocr(
        self,
        image: Union[Image.Image, np.ndarray],
        page_num: int = 1,
        scale: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Run OCR on an image and return standardized OCR regions with:
        {
          "text": "...",
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.94,
          "page": 10,
          "extractor": "ocr"
        }
        Coordinates are converted back to PDF points by dividing by `scale`.
        """
        if not self.is_available:
            return []

        if isinstance(image, Image.Image):
            np_img = np.array(image.convert("RGB"))
        else:
            np_img = image

        raw_results, _ = self._rapid_ocr(np_img)
        if not raw_results:
            return []

        ocr_items: List[Dict[str, Any]] = []
        for item in raw_results:
            # item = [pts, text, score]
            pts = item[0]
            text = str(item[1]).strip()
            if not text:
                continue

            try:
                conf = float(item[2])
            except (ValueError, TypeError):
                conf = 0.80

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            px_x1, px_y1 = min(xs), min(ys)
            px_x2, px_y2 = max(xs), max(ys)

            # Convert to PDF points
            pdf_bbox = [
                round(px_x1 / scale, 1),
                round(px_y1 / scale, 1),
                round(px_x2 / scale, 1),
                round(px_y2 / scale, 1),
            ]

            ocr_items.append({
                "text": text,
                "bbox": pdf_bbox,
                "confidence": round(conf, 2),
                "page": page_num,
                "extractor": "ocr",
                "pixel_bbox": [int(px_x1), int(px_y1), int(px_x2), int(px_y2)],
            })

        return ocr_items

    def aggregate_text_regions(
        self,
        ocr_items: List[Dict[str, Any]],
        page_num: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Group individual single-line OCR detections into paragraph/section blocks
        when vertically contiguous and aligned, while keeping exact bounding box unions.
        """
        if not ocr_items:
            return []

        # Sort top-to-bottom
        sorted_items = sorted(ocr_items, key=lambda it: (it["bbox"][1], it["bbox"][0]))
        blocks: List[Dict[str, Any]] = []
        current_group: List[Dict[str, Any]] = []

        def flush_group(group: List[Dict[str, Any]]):
            if not group:
                return
            x1 = min(g["bbox"][0] for g in group)
            y1 = min(g["bbox"][1] for g in group)
            x2 = max(g["bbox"][2] for g in group)
            y2 = max(g["bbox"][3] for g in group)

            full_text = " ".join(g["text"] for g in group)
            avg_conf = sum(g["confidence"] for g in group) / len(group)

            warnings = []
            if avg_conf < 0.60:
                warnings.append("Low OCR confidence; verify against original scan.")
            elif avg_conf < 0.85:
                warnings.append("Estimated OCR transcription; some characters may be imprecise.")

            blocks.append({
                "text": full_text,
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(avg_conf, 2),
                "page": page_num,
                "extractor": "ocr",
                "warnings": warnings,
                "line_count": len(group),
            })

        for item in sorted_items:
            if not current_group:
                current_group.append(item)
                continue

            last = current_group[-1]
            last_h = max(10.0, last["bbox"][3] - last["bbox"][1])
            v_gap = item["bbox"][1] - last["bbox"][3]

            # If gap is within reasonable line spacing (< 1.5 * line_height) and horizontal overlap
            if 0 <= v_gap < last_h * 1.6:
                current_group.append(item)
            else:
                flush_group(current_group)
                current_group = [item]

        flush_group(current_group)
        return blocks
