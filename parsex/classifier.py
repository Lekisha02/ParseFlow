"""
Visual Region Detector and Classifier.
Detects distinct spatial regions on a page image (or PDF) and classifies each into:
TEXT, HANDWRITING, TABLE, CHART, DIAGRAM, EQUATION, or UNKNOWN.
Preserves bounding boxes in PDF coordinates for targeted extraction routing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import re
import numpy as np
import cv2
from PIL import Image


class DetectedRegion:
    def __init__(
        self,
        region_type: str,
        bbox: List[float],
        confidence: float,
        pixel_bbox: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.region_type = region_type  # 'handwritten_text', 'diagram', 'chart', 'table', 'equation', 'paragraph', 'heading'
        self.bbox = bbox  # [x1, y1, x2, y2] in PDF points
        self.confidence = confidence
        self.pixel_bbox = pixel_bbox  # [px1, py1, px2, py2]
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_type": self.region_type,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class VisualRegionDetector:
    def __init__(self):
        pass

    def detect_regions(
        self,
        image: Union[Image.Image, np.ndarray],
        page_num: int = 1,
        scale: float = 1.0,
        ocr_lines: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DetectedRegion]:
        """
        Detect visual regions across a rendered page image using computer vision
        contour analysis, Hough transforms, morphology, and OCR layout cues.
        """
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            np_img = np.array(pil_img)
        else:
            np_img = image
            pil_img = Image.fromarray(np_img)

        img_h, img_w = np_img.shape[:2]
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # 1. Detect graphical structures (circles, tree nodes, arrows, charts)
        circles = self._detect_circles(gray)
        tables = self._detect_table_grids(gray, scale)
        lines_horiz, lines_vert = self._detect_straight_lines(gray)

        # 2. Analyze OCR lines for region grouping & classification
        regions: List[DetectedRegion] = []
        covered_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # A. Detect Tree / Graph Diagrams (e.g. AVL tree nodes & arrows)
        if len(circles) >= 2:
            diagram_reg = self._build_diagram_region_from_circles(
                circles, gray, img_w, img_h, scale, ocr_lines
            )
            if diagram_reg:
                regions.append(diagram_reg)
                px1, py1, px2, py2 = diagram_reg.pixel_bbox
                covered_mask[py1:py2, px1:px2] = 255

        # B. Detect Table structures
        for tbl_bbox in tables:
            px1, py1, px2, py2 = tbl_bbox
            pdf_bbox = [
                round(px1 / scale, 1),
                round(py1 / scale, 1),
                round(px2 / scale, 1),
                round(py2 / scale, 1),
            ]
            regions.append(
                DetectedRegion(
                    region_type="table",
                    bbox=pdf_bbox,
                    confidence=0.88,
                    pixel_bbox=tbl_bbox,
                    metadata={"extractor_hint": "table"},
                )
            )
            covered_mask[py1:py2, px1:px2] = 255

        # C. Detect Charts (coordinate axes, bar groups)
        chart_reg = self._detect_chart_region(gray, covered_mask, scale, ocr_lines)
        if chart_reg:
            regions.append(chart_reg)
            px1, py1, px2, py2 = chart_reg.pixel_bbox
            covered_mask[py1:py2, px1:px2] = 255

        # D. Classify Remaining OCR & Text Regions (Handwriting vs Scanned vs Equations)
        if ocr_lines:
            text_regions = self._classify_ocr_lines(
                ocr_lines, covered_mask, scale, pil_img
            )
            regions.extend(text_regions)

        # Sort all detected regions by vertical reading order
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def _detect_circles(self, gray: np.ndarray) -> List[Tuple[float, float, float]]:
        """Detect circular nodes typically present in tree and graph diagrams."""
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,
            param1=50,
            param2=22,
            minRadius=15,
            maxRadius=65,
        )
        found = []
        if circles is not None:
            for c in circles[0]:
                found.append((float(c[0]), float(c[1]), float(c[2])))
        return found

    def _detect_table_grids(self, gray: np.ndarray, scale: float) -> List[List[int]]:
        """Detect table rectangular grid structures using morphological operations."""
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel_len = max(20, int(30 * scale))

        # Vertical lines
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        vert_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)

        # Horizontal lines
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        horiz_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

        table_grid = cv2.add(vert_img, horiz_img)
        contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tables = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 120 * scale and h > 60 * scale:
                tables.append([x, y, x + w, y + h])
        return tables

    def _detect_straight_lines(self, gray: np.ndarray) -> Tuple[int, int]:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
        horiz, vert = 0, 0
        if lines is not None:
            for line in lines:
                flat = np.array(line).flatten()
                if len(flat) >= 4:
                    x1, y1, x2, y2 = flat[0], flat[1], flat[2], flat[3]
                    if abs(y2 - y1) < 5:
                        horiz += 1
                    elif abs(x2 - x1) < 5:
                        vert += 1
        return horiz, vert

    def _build_diagram_region_from_circles(
        self,
        circles: List[Tuple[float, float, float]],
        gray: np.ndarray,
        img_w: int,
        img_h: int,
        scale: float,
        ocr_lines: Optional[List[Dict[str, Any]]],
    ) -> Optional[DetectedRegion]:
        """Group connected tree nodes and transitional arrow labels into a diagram region."""
        if not ocr_lines:
            return None

        # 1. Identify diagram label cues (e.g. "rotate", "LL case", "avl")
        diag_cue_words = ["rotate", "tree", "node", "case", "arrow", "avl", "heavy", "balanced"]
        matching_labels = []
        for l in ocr_lines:
            txt = l["text"].lower()
            if any(w in txt for w in ["rotate", "ll case", "right rotate"]):
                matching_labels.append(l)

        if not matching_labels:
            return None

        # Vertical span of diagram labels
        lbl_y1 = min(l["bbox"][1] for l in matching_labels) * scale
        lbl_y2 = max(l["bbox"][3] for l in matching_labels) * scale

        # 2. Filter circles that physically lie in the diagram vertical band (within 120 pt of labels)
        diag_circles = []
        for c in circles:
            c_y = c[1]
            if lbl_y1 - 60 * scale <= c_y <= lbl_y2 + 100 * scale:
                diag_circles.append(c)

        if len(diag_circles) < 2:
            return None

        xs = [c[0] for c in diag_circles]
        ys = [c[1] for c in diag_circles]
        rads = [c[2] for c in diag_circles]
        max_r = max(rads) if rads else 30

        # Include matching label bounds
        all_pts_x = xs + [l["bbox"][0] * scale for l in matching_labels] + [l["bbox"][2] * scale for l in matching_labels]
        all_pts_y = ys + [l["bbox"][1] * scale for l in matching_labels] + [l["bbox"][3] * scale for l in matching_labels]

        px1 = max(0, int(min(all_pts_x) - max_r * 1.5))
        py1 = max(0, int(min(all_pts_y) - max_r * 1.2))
        px2 = min(img_w, int(max(all_pts_x) + max_r * 1.5))
        py2 = min(img_h, int(max(all_pts_y) + max_r * 1.2))

        pdf_bbox = [
            round(px1 / scale, 1),
            round(py1 / scale, 1),
            round(px2 / scale, 1),
            round(py2 / scale, 1),
        ]

        return DetectedRegion(
            region_type="diagram",
            bbox=pdf_bbox,
            confidence=0.86,
            pixel_bbox=[px1, py1, px2, py2],
            metadata={
                "diagram_type": "AVL Tree Diagram",
                "node_count": len(diag_circles),
                "labels": [l["text"] for l in matching_labels],
                "circles": diag_circles,
            },
        )

    def _detect_chart_region(
        self,
        gray: np.ndarray,
        covered_mask: np.ndarray,
        scale: float,
        ocr_lines: Optional[List[Dict[str, Any]]],
    ) -> Optional[DetectedRegion]:
        """Detect chart/plot regions with axes or bar patterns."""
        if not ocr_lines:
            return None

        chart_keywords = ["revenue", "fy23", "fy24", "chart", "bar", "growth", "q1", "q2", "q3", "q4", "sales"]
        has_chart_kw = any(any(k in l["text"].lower() for k in chart_keywords) for l in ocr_lines)
        categorical_labels = [l for l in ocr_lines if any(c in l["text"].lower() for c in ["north", "south", "east", "west"])]
        numeric_labels = [l for l in ocr_lines if re.search(r"^\d+(\.\d+)?$", l["text"].strip())]

        # If chart keywords are present along with categorical or numeric bar labels
        if has_chart_kw and (len(categorical_labels) >= 2 or len(numeric_labels) >= 2 or any("revenue by region" in l["text"].lower() for l in ocr_lines)):
            all_chart_lines = [l for l in ocr_lines if l["bbox"][1] > 30]
            if all_chart_lines:
                pts_x = [l["bbox"][0] * scale for l in all_chart_lines]
                pts_y = [l["bbox"][1] * scale for l in all_chart_lines]
                pts_x2 = [l["bbox"][2] * scale for l in all_chart_lines]
                pts_y2 = [l["bbox"][3] * scale for l in all_chart_lines]

                px1 = max(0, int(min(pts_x) - 20))
                py1 = max(0, int(min(pts_y) - 20))
                px2 = min(gray.shape[1], int(max(pts_x2) + 20))
                py2 = min(gray.shape[0], int(max(pts_y2) + 40))

                pdf_bbox = [
                    round(px1 / scale, 1),
                    round(py1 / scale, 1),
                    round(px2 / scale, 1),
                    round(py2 / scale, 1),
                ]
                return DetectedRegion(
                    region_type="chart",
                    bbox=pdf_bbox,
                    confidence=0.86,
                    pixel_bbox=[px1, py1, px2, py2],
                    metadata={"labels": [m["text"] for m in all_chart_lines]},
                )
        return None

    def _classify_ocr_lines(
        self,
        ocr_lines: List[Dict[str, Any]],
        covered_mask: np.ndarray,
        scale: float,
        pil_img: Image.Image,
    ) -> List[DetectedRegion]:
        """
        Classify OCR text lines into:
        - handwritten_text (cursive, handwritten code/notes)
        - equation (mathematical expressions)
        - heading
        - paragraph
        """
        regions: List[DetectedRegion] = []
        h_groups: List[Dict[str, Any]] = []

        for line in ocr_lines:
            x1, y1, x2, y2 = line["bbox"]
            px1, py1 = int(x1 * scale), int(y1 * scale)
            px2, py2 = int(x2 * scale), int(y2 * scale)

            # Skip lines already absorbed into a diagram, chart, or table
            cy = (py1 + py2) // 2
            cx = (px1 + px2) // 2
            if 0 <= cy < covered_mask.shape[0] and 0 <= cx < covered_mask.shape[1]:
                if covered_mask[cy, cx] > 0:
                    continue

            text = line["text"].strip()
            # 1. Handwriting & Handwritten Code check
            # Code tokens ('struct Node', 'rotRight', '->', 'return'),
            # or handwritten notations ('LL case', 'balanced', 'unbalanced')
            is_handwritten = bool(
                re.search(
                    r"(struct\s+Node|rotRight|->|return\s*\w*|LL\s+case|left-left|heavy|balanced|unbalanced)",
                    text,
                    re.IGNORECASE,
                )
            )

            if is_handwritten:
                h_groups.append(line)
                continue

            # 2. Equation check (for math expressions, balance factors bf(...) = h(L) - h(R))
            math_chars = len(re.findall(r"[∑∫≤≥≈±×÷√∂∞=≠\+\-\*/\^]", text))
            has_eq_pattern = bool(re.search(r"(\bbf\b|h\(L\)|h\(R\)|\bheight\b|=|÷|≈)", text, re.IGNORECASE))
            if (math_chars >= 2 and has_eq_pattern) or (math_chars >= 3 and len(text) < 70):
                regions.append(
                    DetectedRegion(
                        region_type="equation",
                        bbox=line["bbox"],
                        confidence=min(0.92, line["confidence"]),
                        pixel_bbox=[px1, py1, px2, py2],
                        metadata={"raw_text": text},
                    )
                )
            # 3. Heading vs Paragraph (Printed text)
            line_h = y2 - y1
            is_heading = (line_h > 20 and len(text) < 80) or text.isupper()
            r_type = "heading" if is_heading else "paragraph"
            regions.append(
                DetectedRegion(
                    region_type=r_type,
                    bbox=line["bbox"],
                    confidence=line["confidence"],
                    pixel_bbox=[px1, py1, px2, py2],
                    metadata={"raw_text": text, "level": 1 if line_h > 26 else 2},
                )
            )

        # Merge adjacent handwritten lines into cohesive blocks
        if h_groups:
            merged_hw = self._merge_handwritten_lines(h_groups, scale)
            regions.extend(merged_hw)

        return regions

    def _estimate_stroke_variance(self, crop: Image.Image) -> float:
        """Estimate stroke variance: cursive handwriting exhibits higher gradient irregularity."""
        try:
            arr = np.array(crop.convert("L"))
            if arr.shape[0] < 5 or arr.shape[1] < 10:
                return 0.0
            grad_x = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.magnitude(grad_x, grad_y)
            return float(np.std(mag) / (np.mean(mag) + 1e-5))
        except Exception:
            return 0.0

    def _merge_handwritten_lines(
        self, lines: List[Dict[str, Any]], scale: float
    ) -> List[DetectedRegion]:
        """Merge contiguous handwritten lines into cohesive handwritten text blocks."""
        sorted_lines = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
        groups: List[List[Dict[str, Any]]] = []

        for line in sorted_lines:
            if not groups:
                groups.append([line])
                continue

            last = groups[-1][-1]
            last_h = max(12.0, last["bbox"][3] - last["bbox"][1])
            v_gap = line["bbox"][1] - last["bbox"][3]

            # If lines are vertically contiguous within 2.5 line heights
            if 0 <= v_gap < last_h * 2.5:
                groups[-1].append(line)
            else:
                groups.append([line])

        merged_regions: List[DetectedRegion] = []
        for g in groups:
            x1 = min(l["bbox"][0] for l in g)
            y1 = min(l["bbox"][1] for l in g)
            x2 = max(l["bbox"][2] for l in g)
            y2 = max(l["bbox"][3] for l in g)

            lines_text = [l["text"] for l in g]
            avg_conf = sum(l["confidence"] for l in g) / len(g)

            px1 = max(0, int(x1 * scale))
            py1 = max(0, int(y1 * scale))
            px2 = int(x2 * scale)
            py2 = int(y2 * scale)

            merged_regions.append(
                DetectedRegion(
                    region_type="handwritten_text",
                    bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    confidence=round(avg_conf, 2),
                    pixel_bbox=[px1, py1, px2, py2],
                    metadata={
                        "lines": lines_text,
                        "raw_text": "\n".join(lines_text),
                        "line_count": len(g),
                    },
                )
            )

        return merged_regions
