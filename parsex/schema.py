"""
ParseX Canonical Block Schema and Document Normalization.
Ensures every extracted element preserves bounding boxes, confidence,
extractor provenance, verification status, and structured content.
"""

from typing import Any, Dict, List, Optional, Union
import json


def create_block(
    block_type: str,
    page: int,
    bbox: Optional[List[float]],
    content: Any,
    confidence: float,
    extractor: str,
    warnings: Optional[List[str]] = None,
    level: Optional[int] = None,
    estimated: bool = False,
    vision: bool = False,
    image_crop_b64: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a canonical ParseX block dictionary."""
    rounded_bbox = [round(float(v), 1) for v in bbox] if bbox else None
    warn_list = list(warnings) if warnings else []
    conf = round(max(0.0, min(1.0, float(confidence))), 2)

    block: Dict[str, Any] = {
        "block_id": "",
        "type": block_type,
        "page": page,
        "bbox": rounded_bbox,
        "reading_order": 0,
        "content": content,
        "confidence": conf,
        "extractor": extractor,
        "warnings": warn_list,
        "estimated": estimated,
        "vision": vision,
        "relationships": [],
    }

    if level is not None:
        block["level"] = level
    if image_crop_b64:
        block["image_crop_b64"] = image_crop_b64
    if extra:
        block.update(extra)

    block["verification"] = compute_verification(block)
    return block


def compute_verification(block: Dict[str, Any]) -> str:
    """
    Truthful verification rule:
    - 'failed': confidence < 0.40
    - 'low_confidence': confidence < 0.85, estimated confidence, or presence of warnings
    - 'verified': confidence >= 0.85 with no warnings and deterministic extraction
    """
    conf = block.get("confidence", 0.0)
    warnings = block.get("warnings", [])
    estimated = block.get("estimated", False)

    if conf < 0.40:
        return "failed"
    if estimated or conf < 0.85 or len(warnings) > 0:
        return "low_confidence"
    return "verified"


def normalize_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign reading order, block IDs, and compute document-level relationships."""
    # Sort blocks by page, then top-to-bottom, left-to-right
    blocks.sort(
        key=lambda b: (
            b.get("page", 1),
            b["bbox"][1] if b.get("bbox") else 0,
            b["bbox"][0] if b.get("bbox") else 0,
        )
    )

    last_heading = ""
    for idx, b in enumerate(blocks, start=1):
        b["block_id"] = f"block_{idx:02d}"
        b["reading_order"] = idx
        b["verification"] = compute_verification(b)

        if b["type"] == "heading":
            last_heading = b.get("content", "")
        elif b["type"] == "table" and not b.get("title") and last_heading:
            b["title"] = last_heading

    return blocks


def format_markdown_table(rows: List[List[Any]]) -> str:
    """Format a 2D list into Markdown table syntax."""
    if not rows or not rows[0]:
        return ""
    col_count = max(len(r) for r in rows)
    padded_rows = [r + [""] * (col_count - len(r)) for r in rows]

    def esc_cell(val: Any) -> str:
        return str(val).replace("|", "\\|").replace("\n", " ").strip()

    header = "| " + " | ".join(esc_cell(c) for c in padded_rows[0]) + " |"
    divider = "| " + " | ".join(["---"] * col_count) + " |"
    body = [
        "| " + " | ".join(esc_cell(c) for c in r) + " |" for r in padded_rows[1:]
    ]
    return header + "\n" + divider + ("\n" + "\n".join(body) if body else "")


def block_to_markdown(b: Dict[str, Any], include_provenance: bool = True) -> str:
    """Convert an individual block to rich, faithful Markdown."""
    b_type = b.get("type", "paragraph")
    content = b.get("content", "")
    page = b.get("page", 1)
    bbox = b.get("bbox", [])
    conf = b.get("confidence", 0.0)
    extractor = b.get("extractor", "unknown")
    verification = b.get("verification", "verified")
    warnings = b.get("warnings", [])

    lines: List[str] = []

    # Flag warning callout if not verified
    if verification != "verified":
        warn_text = " ".join(warnings) if warnings else "Low confidence extraction."
        tag = "⚠ LOW CONFIDENCE" if verification == "low_confidence" else "✕ FAILED"
        lines.append(f"> {tag} ({conf}): {warn_text}\n")

    if b_type == "heading":
        lvl = b.get("level", 2)
        lines.append(f"{'#' * lvl} {content}\n")

    elif b_type == "paragraph":
        lines.append(f"{content}\n")

    elif b_type == "list":
        if isinstance(content, list):
            lines.append("\n".join(f"- {item}" for item in content) + "\n")
        else:
            lines.append(f"- {content}\n")

    elif b_type == "table":
        if isinstance(content, list) and content:
            lines.append(format_markdown_table(content) + "\n")
        else:
            lines.append(f"*(Table on page {page})*\n")

    elif b_type == "equation":
        lines.append(f"$$\n{content}\n$$\n")

    elif b_type == "handwritten_text":
        lines.append(f"### Handwritten Text (Page {page})\n")
        if isinstance(content, str):
            lines.append(f"```\n{content}\n```\n")
        elif isinstance(content, dict):
            if content.get("description"):
                lines.append(f"{content['description']}\n")
            if content.get("transcription"):
                lines.append(f"```\n{content['transcription']}\n```\n")

    elif b_type in ("diagram", "chart", "figure"):
        title_str = ""
        if isinstance(content, dict):
            title_str = content.get("title", "")
        hdr = b_type.capitalize()
        if title_str:
            lines.append(f"### {hdr}: {title_str}\n")
        else:
            lines.append(f"### {hdr} (Page {page})\n")

        lines.append(f"**Source:** Page {page}, bbox {json.dumps(bbox)} · confidence {conf} · {extractor}\n")

        if isinstance(content, dict):
            if content.get("description"):
                lines.append(f"{content['description']}\n")
            if content.get("transcription"):
                lines.append(f"```\n{content['transcription']}\n```\n")
            if content.get("elements"):
                lines.append("**Elements:**\n" + "\n".join(f"- {e}" for e in content["elements"]) + "\n")
            if content.get("relationships"):
                lines.append("**Relationships:**\n" + "\n".join(f"- {r}" for r in content["relationships"]) + "\n")
            if content.get("data"):
                lines.append(f"**Data (as read):** `{json.dumps(content['data'])}`\n")
        elif isinstance(content, str) and content:
            lines.append(f"*{content}*\n")
        else:
            lines.append(f"*[Visual {b_type} preserved]*\n")

    else:
        lines.append(f"{content}\n")

    if include_provenance:
        status_label = "estimated confidence" if b.get("estimated") else "confidence"
        prov = f"<!-- {b.get('block_id')} · page {page} · bbox {json.dumps(bbox)} · {status_label} {conf} · {extractor} -->\n"
        lines.append(prov)

    return "\n".join(lines).strip() + "\n\n"


def document_to_markdown(doc: Dict[str, Any], include_provenance: bool = True) -> str:
    """Generate complete document Markdown."""
    md_parts = []
    filename = doc.get("filename", "Document")
    md_parts.append(f"# {filename}\n\n")

    for block in doc.get("blocks", []):
        md_parts.append(block_to_markdown(block, include_provenance))

    return "".join(md_parts).strip() + "\n"
