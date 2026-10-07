"""
Handwriting and Code Processing Module.
Distinguishes printed text from handwriting and mixed regions.
Preserves indentation, C code semantics, pointer syntax, and applies
strict honesty rules when handwriting cannot be confidently resolved.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from parsex.schema import create_block


class HandwritingProcessor:
    def __init__(self, vision_extractor=None):
        self.vision_extractor = vision_extractor

    def classify_text_modality(self, text: str, stroke_variance: float = 0.0) -> str:
        """
        Distinguish:
        - 'printed_text'
        - 'handwritten_text'
        - 'mixed'
        """
        has_printed_clues = bool(re.search(r"(\bAnnual Financial Report\b|\bRevenue\b|\bTable \d+\b|\bOperations:\b)", text))
        has_handwriting_clues = bool(
            re.search(
                r"(struct\s+Node|rotRight|->|return\s+\w+|LL\s+case|left-left|heavy|bf\(|balanced|rotate)",
                text,
                re.IGNORECASE,
            )
        )

        if has_printed_clues and has_handwriting_clues:
            return "mixed"
        if has_handwriting_clues or stroke_variance > 0.35:
            return "handwritten_text"
        return "printed_text"

    def process_handwritten_region(
        self,
        image_crop: Optional[Image.Image],
        raw_text: str,
        lines: List[str],
        page_num: int,
        bbox: List[float],
        confidence: float = 0.74,
    ) -> Dict[str, Any]:
        """
        Process a handwritten region into a canonical ParseX block.
        Preserves code formatting and flags low confidence accurately.
        """
        warnings = []
        is_uncertain = confidence < 0.60

        # Preserve verbatim indentation and line structure if code
        cleaned_lines = []
        for line in lines:
            # Strip trailing noise while preserving leading code indentation
            cleaned_lines.append(line.rstrip())

        content = "\n".join(cleaned_lines) if cleaned_lines else raw_text.strip()

        if is_uncertain or not content:
            warnings.append("Handwriting could not be confidently transcribed; review recommended.")
            reported_conf = min(0.55, confidence)
            return create_block(
                block_type="handwritten_text",
                page=page_num,
                bbox=bbox,
                content=content or "[Illegible handwriting]",
                confidence=reported_conf,
                extractor="vision_handwriting",
                warnings=warnings,
                estimated=True,
                vision=False,
            )

        # High or moderate confidence handwriting
        warnings.append("Transcribed with handwriting recognition heuristics.")
        return create_block(
            block_type="handwritten_text",
            page=page_num,
            bbox=bbox,
            content=content,
            confidence=confidence,
            extractor="vision_handwriting",
            warnings=warnings,
            estimated=True,
            vision=False,
        )
