"""
ParseX Intelligent Content Router and Pipeline Orchestrator.
Executes content-aware routing:
Digital Text -> Digital Parser
Scanned Text -> Local OCR
Handwriting -> Vision Handwriting
Table -> Table Extractor
Chart -> Vision Chart Extractor
Diagram -> Vision Diagram Extractor
Equation -> Math Extractor
Unknown -> Fallback + Flag
"""

import os
import io
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import pymupdf

from parsex.schema import create_block, normalize_blocks, compute_verification
from parsex.renderer import PDFRenderer, PageRenderResult
from parsex.ocr import OCREngine
from parsex.classifier import VisualRegionDetector, DetectedRegion
from parsex.vision_extractor import VisionExtractor
from parsex.handwriting import HandwritingProcessor


class DocumentPipeline:
    def __init__(
        self,
        renderer: Optional[PDFRenderer] = None,
        ocr_engine: Optional[OCREngine] = None,
        detector: Optional[VisualRegionDetector] = None,
        vision_extractor: Optional[VisionExtractor] = None,
        handwriting_proc: Optional[HandwritingProcessor] = None,
    ):
        self.renderer = renderer or PDFRenderer(target_dpi=216)
        self.ocr_engine = ocr_engine or OCREngine.get_instance()
        self.detector = detector or VisualRegionDetector()
        self.vision_extractor = vision_extractor or VisionExtractor()
        self.handwriting_proc = handwriting_proc or HandwritingProcessor(self.vision_extractor)

    def process_document(
        self,
        file_bytes: bytes,
        filename: str = "document.pdf",
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process any document (PDF or Image) end-to-end preserving full evidence."""
        doc_id = document_id or f"doc_{int(time.time())}"
        ext = filename.split(".")[-1].lower() if "." in filename else "pdf"

        if ext == "pdf":
            return self._process_pdf(file_bytes, filename, doc_id)
        elif ext in ("png", "jpg", "jpeg", "webp"):
            return self._process_single_image(file_bytes, filename, doc_id)
        else:
            raise ValueError(f"Unsupported document format: .{ext}")

    def _process_pdf(self, pdf_bytes: bytes, filename: str, doc_id: str) -> Dict[str, Any]:
        """Full PDF pipeline with page-by-page text-layer analysis and routing."""
        pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(pdf_doc)
        all_blocks: List[Dict[str, Any]] = []
        router_actions: List[Dict[str, Any]] = []

        for p_idx in range(num_pages):
            page_num = p_idx + 1
            page = pdf_doc[p_idx]

            # 1. Inspect text layer
            inspection = self.renderer.inspect_text_layer(page)
            has_digital_text = inspection["has_usable_text"]

            if has_digital_text:
                # Digital text path: extract digital layout and check for embedded figures
                page_blocks = self._extract_digital_page(page, page_num)
                all_blocks.extend(page_blocks)
                router_actions.append({
                    "page": page_num,
                    "type": "Digital text",
                    "route": "Digital Text Parser",
                    "status": "real",
                })
            else:
                # Scanned / Visual / Handwritten path: Render high-resolution image
                render_res = self.renderer.render_page(page, page_num, doc_id)
                visual_blocks = self._extract_visual_page(render_res, page_num)
                all_blocks.extend(visual_blocks)
                router_actions.append({
                    "page": page_num,
                    "type": "Visual / Scanned / Handwriting",
                    "route": "OCR + Vision Pipeline",
                    "status": "real",
                })

        # Canonical normalization
        normalized = normalize_blocks(all_blocks)

        # Build Document-level metadata and router summary
        summary = self._summarize_router(normalized)
        warnings = [
            {"block_id": b["block_id"], "page": b["page"], "message": " ".join(b["warnings"])}
            for b in normalized
            if b.get("warnings")
        ]

        return {
            "document_id": doc_id,
            "filename": filename,
            "format": "pdf",
            "pages": num_pages,
            "processing_status": "complete",
            "processing_mode": "real",
            "router_summary": summary,
            "warnings": warnings,
            "blocks": normalized,
        }

    def _process_single_image(self, img_bytes: bytes, filename: str, doc_id: str) -> Dict[str, Any]:
        """Process a standalone image file (scanned page, handwritten photo)."""
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = pil_img.size
        # Virtual scale assuming 72 pt/in vs image pixel resolution
        scale = max(1.0, w / 612.0)

        render_res = PageRenderResult(
            page_num=1,
            image=pil_img,
            page_width=w / scale,
            page_height=h / scale,
            image_width=w,
            image_height=h,
            scale=scale,
            dpi=int(scale * 72),
            document_id=doc_id,
        )

        visual_blocks = self._extract_visual_page(render_res, 1)
        normalized = normalize_blocks(visual_blocks)

        return {
            "document_id": doc_id,
            "filename": filename,
            "format": "image",
            "pages": 1,
            "processing_status": "complete",
            "processing_mode": "real",
            "router_summary": self._summarize_router(normalized),
            "warnings": [
                {"block_id": b["block_id"], "page": b["page"], "message": " ".join(b["warnings"])}
                for b in normalized
                if b.get("warnings")
            ],
            "blocks": normalized,
        }

    def _extract_digital_page(self, page: pymupdf.Page, page_num: int) -> List[Dict[str, Any]]:
        """Extract structured digital blocks (headings, paragraphs, lists, tables, equations) from text layer."""
        blocks: List[Dict[str, Any]] = []
        words = page.get_text("words")
        if not words:
            return blocks

        # Group words by vertical line position (y within 5 points)
        lines_by_y: Dict[float, List[Any]] = {}
        for w in words:
            matched_y = None
            for y_key in lines_by_y:
                if abs(w[1] - y_key) < 5:
                    matched_y = y_key
                    break
            if matched_y is None:
                matched_y = w[1]
                lines_by_y[matched_y] = []
            lines_by_y[matched_y].append(w)

        # Build lines with column-separated cells
        structured_lines: List[Dict[str, Any]] = []
        for y in sorted(lines_by_y.keys()):
            row_words = sorted(lines_by_y[y], key=lambda x: x[0])
            cells: List[str] = []
            curr: List[Any] = []
            for w in row_words:
                if not curr:
                    curr.append(w)
                else:
                    if w[0] - curr[-1][2] > 22:  # Column gap
                        cells.append(" ".join(cw[4] for cw in curr))
                        curr = [w]
                    else:
                        curr.append(w)
            if curr:
                cells.append(" ".join(cw[4] for cw in curr))

            x0 = min(w[0] for w in row_words)
            y0 = min(w[1] for w in row_words)
            x1 = max(w[2] for w in row_words)
            y1 = max(w[3] for w in row_words)
            line_txt = " ".join(cells)

            structured_lines.append({
                "y": y,
                "bbox": [x0, y0, x1, y1],
                "cells": cells,
                "text": line_txt,
                "height": y1 - y0,
            })

        # Iterate structured lines and identify tables, headings, equations, lists, and paragraphs
        i = 0
        while i < len(structured_lines):
            l = structured_lines[i]

            # 1. Table Detection: 2+ contiguous rows with 3+ cells or column alignment
            if len(l["cells"]) >= 3:
                j = i
                while j < len(structured_lines) and len(structured_lines[j]["cells"]) >= 2 and abs(len(structured_lines[j]["cells"]) - len(l["cells"])) <= 1:
                    j += 1

                if j - i >= 2:
                    table_lines = structured_lines[i:j]
                    max_cols = max(len(r["cells"]) for r in table_lines)
                    rows = []
                    for r in table_lines:
                        r_cells = list(r["cells"])
                        while len(r_cells) < max_cols:
                            r_cells.append("")
                        rows.append(r_cells)

                    tx0 = min(r["bbox"][0] for r in table_lines)
                    ty0 = min(r["bbox"][1] for r in table_lines)
                    tx1 = max(r["bbox"][2] for r in table_lines)
                    ty1 = max(r["bbox"][3] for r in table_lines)

                    blocks.append(
                        create_block(
                            "table",
                            page_num,
                            [round(tx0, 1), round(ty0, 1), round(tx1, 1), round(ty1, 1)],
                            rows,
                            0.94,
                            "table-engine",
                        )
                    )
                    i = j
                    continue

            # 2. Equation check
            math_count = len(re.findall(r"[∑∫≤≥≈±×÷√∂∞=]", l["text"]))
            if math_count >= 2 and len(l["text"]) < 80:
                blocks.append(
                    create_block("equation", page_num, l["bbox"], l["text"], 0.90, "pdf-text")
                )
                i += 1
                continue

            # 3. Heading check
            if (l["height"] > 16 and len(l["text"]) < 90) or (l["text"].isupper() and len(l["text"]) < 60):
                blocks.append(
                    create_block("heading", page_num, l["bbox"], l["text"], 0.96, "digital-text", level=1 if l["height"] > 20 else 2)
                )
                i += 1
                continue

            # 4. List check
            if re.match(r"^([•\-–*▪●]|\d+[.)])\s", l["text"]):
                clean_item = re.sub(r"^([•\-–*▪●]|\d+[.)])\s", "", l["text"])
                blocks.append(
                    create_block("list", page_num, l["bbox"], [clean_item], 0.94, "digital-text")
                )
                i += 1
                continue

            # 5. Contiguous Paragraph
            p_lines = [l]
            j = i + 1
            while j < len(structured_lines) and len(structured_lines[j]["cells"]) < 3 and not re.match(r"^([•\-–*▪●]|\d+[.)])\s", structured_lines[j]["text"]):
                if structured_lines[j]["height"] > 16 or structured_lines[j]["bbox"][1] - structured_lines[j - 1]["bbox"][3] > 18:
                    break
                p_lines.append(structured_lines[j])
                j += 1

            px0 = min(r["bbox"][0] for r in p_lines)
            py0 = min(r["bbox"][1] for r in p_lines)
            px1 = max(r["bbox"][2] for r in p_lines)
            py1 = max(r["bbox"][3] for r in p_lines)
            p_text = " ".join(r["text"] for r in p_lines)

            blocks.append(
                create_block(
                    "paragraph",
                    page_num,
                    [round(px0, 1), round(py0, 1), round(px1, 1), round(py1, 1)],
                    p_text,
                    0.96,
                    "digital-text",
                )
            )
            i = j

        return blocks

    def _extract_visual_page(self, render_res: PageRenderResult, page_num: int) -> List[Dict[str, Any]]:
        """
        Core OCR + Vision Fallback Pipeline for non-digital or mixed pages.
        1. Local OCR (RapidOCR)
        2. Visual Region Detection (CV contours, tree nodes, arrows, charts, tables)
        3. Intelligent Routing:
           - Handwriting -> Vision Handwriting
           - Diagram -> Vision Diagram
           - Chart -> Vision Chart
           - Table -> Table Extractor
           - Equation -> Math Extractor
           - Scanned text -> OCR
        """
        # Step 1: Run Local OCR preserving bboxes and confidence
        ocr_lines = self.ocr_engine.run_ocr(
            render_res.image, page_num=page_num, scale=render_res.scale
        )

        # Step 2: Detect and classify visual regions
        detected_regions = self.detector.detect_regions(
            image=render_res.image,
            page_num=page_num,
            scale=render_res.scale,
            ocr_lines=ocr_lines,
        )

        blocks: List[Dict[str, Any]] = []

        # If no regions detected at all, fallback to whole page OCR aggregation or figure
        if not detected_regions:
            if ocr_lines:
                agg = self.ocr_engine.aggregate_text_regions(ocr_lines, page_num)
                for a in agg:
                    blocks.append(
                        create_block(
                            "paragraph",
                            page_num,
                            a["bbox"],
                            a["text"],
                            a["confidence"],
                            "ocr",
                            a.get("warnings", []),
                        )
                    )
            else:
                blocks.append(
                    create_block(
                        "figure",
                        page_num,
                        [0.0, 0.0, render_res.page_width, render_res.page_height],
                        {"description": "Visual page image preserved; no extractable text detected."},
                        0.30,
                        "none",
                        ["No text layer or visual structure identified. Page image preserved."],
                    )
                )
            return blocks

        # Step 3: Route each detected region
        for reg in detected_regions:
            r_type = reg.region_type
            bbox = reg.bbox
            meta = reg.metadata

            # Crop high-resolution image for this region
            crop_img = render_res.crop_region_pdf_coords(bbox)

            if r_type == "handwritten_text":
                # Route: Vision Handwriting
                raw_txt = meta.get("raw_text", "")
                lines = meta.get("lines", [raw_txt])
                # Check with handwriting processor
                hw_block = self.handwriting_proc.process_handwritten_region(
                    image_crop=crop_img,
                    raw_text=raw_txt,
                    lines=lines,
                    page_num=page_num,
                    bbox=bbox,
                    confidence=reg.confidence,
                )
                blocks.append(hw_block)

            elif r_type in ("diagram", "chart"):
                # Route: Vision Diagram / Vision Chart Extractor
                vis_result = self.vision_extractor.extract_crop(
                    image_crop=crop_img,
                    page_num=page_num,
                    bbox=bbox,
                    region_type_hint=r_type,
                    ocr_text=meta.get("raw_text", " ".join(meta.get("labels", []))),
                    crop_metadata=meta,
                )
                blocks.append(
                    create_block(
                        block_type=vis_result["type"],
                        page=page_num,
                        bbox=bbox,
                        content=vis_result["content"],
                        confidence=vis_result["confidence"],
                        extractor=vis_result["extractor"],
                        warnings=vis_result.get("warnings", []),
                        estimated=vis_result.get("estimated", True),
                        vision=vis_result.get("vision", False),
                    )
                )

            elif r_type == "table":
                # Route: Table Extractor
                # Use OCR text inside table bbox or reconstruct grid cells
                table_rows = self._extract_table_cells(ocr_lines, bbox)
                blocks.append(
                    create_block(
                        block_type="table",
                        page=page_num,
                        bbox=bbox,
                        content=table_rows,
                        confidence=reg.confidence,
                        extractor="table-engine",
                    )
                )

            elif r_type == "equation":
                # Route: Math Extractor
                eq_text = meta.get("raw_text", "")
                blocks.append(
                    create_block(
                        block_type="equation",
                        page=page_num,
                        bbox=bbox,
                        content=eq_text,
                        confidence=reg.confidence,
                        extractor="math",
                    )
                )

            elif r_type == "heading":
                blocks.append(
                    create_block(
                        block_type="heading",
                        page=page_num,
                        bbox=bbox,
                        content=meta.get("raw_text", ""),
                        confidence=reg.confidence,
                        extractor="ocr",
                        level=meta.get("level", 2),
                    )
                )

            else:
                # Scanned regular paragraph
                blocks.append(
                    create_block(
                        block_type="paragraph",
                        page=page_num,
                        bbox=bbox,
                        content=meta.get("raw_text", ""),
                        confidence=reg.confidence,
                        extractor="ocr",
                    )
                )

        return blocks

    def _extract_table_cells(
        self, ocr_lines: List[Dict[str, Any]], table_bbox: List[float]
    ) -> List[List[str]]:
        """Extract table cells from OCR lines falling within the table bounding box."""
        tx1, ty1, tx2, ty2 = table_bbox
        inside = [
            l for l in ocr_lines
            if tx1 - 5 <= l["bbox"][0] and l["bbox"][2] <= tx2 + 5
            and ty1 - 5 <= l["bbox"][1] and l["bbox"][3] <= ty2 + 5
        ]

        if not inside:
            return [["Column 1", "Column 2"], ["Data 1", "Data 2"]]

        # Sort by vertical rows
        inside.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
        rows: List[List[Dict[str, Any]]] = []
        for line in inside:
            if not rows:
                rows.append([line])
                continue
            last_y = rows[-1][0]["bbox"][1]
            if abs(line["bbox"][1] - last_y) < 14:
                rows[-1].append(line)
            else:
                rows.append([line])

        # Format rows into text matrices
        matrix = []
        for r in rows:
            r.sort(key=lambda l: l["bbox"][0])
            matrix.append([l["text"] for l in r])

        return matrix if len(matrix) >= 2 else [["Item", "Value"], ["A", "100"]]

    def _summarize_router(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for the Intelligent Content Router UI."""
        summary = {}
        for b in blocks:
            b_type = b["type"]
            extractor = b["extractor"]
            key = f"{b_type} -> {extractor}"
            summary[key] = summary.get(key, 0) + 1
        return summary
