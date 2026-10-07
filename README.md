# ParseX: Evidence-Preserving Document Intelligence Engine

ParseX converts complex documents (digital text, scanned pages, handwriting, code, diagrams, charts, tables, equations) into structured, verifiable data while preserving complete spatial and evidentiary provenance.

---

## Architecture Overview

```
                      Document (PDF / Image / Office)
                                    │
                                    ▼
                          Page-Level Inspection
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
        [Digital Text Layer]                   [Missing / Scanned / Image]
               │                                         │
       Digital Text Parser                     High-Res Rendering (PyMuPDF)
               │                                   216–300 DPI Pixmap
               │                                         │
               │                               RapidOCR (PP-OCRv4 ONNX)
               │                                         │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
                        Visual Region Detection & Routing
                                    │
       ┌───────────┬────────────┬───┴────────┬───────────┬───────────┐
       ▼           ▼            ▼            ▼           ▼           ▼
    Printed     Scanned     Handwriting   Diagram      Chart       Table
     Text        Text       (Code/Notes)  (Tree/Graph) (Plot)     (Grid/Cells)
       │           │            │            │           │           │
    Digital      Local        Vision       Vision      Vision      Table
    Parser        OCR       Handwriting   Diagram      Chart      Engine
       │           │            │            │           │           │
       └───────────┴────────────┴───┬────────┴───────────┴───────────┘
                                    │
                                    ▼
                       ParseX Canonical Block Schema
                                    │
                                    ▼
                         Verification & Provenance
                     (Honest confidence, Bounding boxes)
                                    │
                                    ▼
                       Outputs: JSON + Markdown + Q&A
```

---

## Fallback Pipeline Architecture

ParseX implements a strict three-tier fallback architecture:
1. **Cloud Multimodal Vision API**: If `VISION_API_KEY` is provided, complex visual regions (diagrams, charts, handwriting) are routed to Google Gemini Multimodal API (`gemini-2.5-flash` or `gemini-1.5-flash`) for deep semantic decomposition.
2. **Local OCR & CV Fallback Engine**: If no API key is provided, the API key is invalid, or the system is offline, ParseX automatically executes its local computer vision pipeline (RapidOCR PP-OCRv4 + OpenCV contour/circle/geometry heuristics + handwriting code reconstruction).
3. **Preserved Raw Visual Fallback**: If a visual region cannot be resolved even by local OCR, the original image crop is preserved with honest confidence (e.g. `0.20-0.40`), `extractor: "none"`, and flagged for human review. **The document or figure never disappears.**

---

## 8 Essential Questions Answered

### 1. What OCR engine was implemented?
**RapidOCR** with **PP-OCRv4 ONNX** model architecture (`rapidocr-onnxruntime`).
- Preserves exact polygon and bounding box coordinates `[x1, y1, x2, y2]`.
- Provides per-line and per-block confidence scores.
- Runs locally without requiring CUDA, Tesseract system binaries, or heavy compiler chains.

### 2. What vision model was implemented, if any?
- **Cloud Multimodal Vision**: Native Google Gemini Multimodal REST API integration supporting `gemini-2.5-flash` and `gemini-1.5-flash`.
- **Local Fallback Vision**: OpenCV contour analysis, Hough transform circle/node detector, directional transition arrow tracker, and domain tree/chart/equation synthesizers.

### 3. What dependencies were added?
- `pymupdf` (v1.28.2) — high-resolution 200–300 DPI PDF pixmap rendering & text-layer inspection
- `rapidocr-onnxruntime` (v1.2.3) — PP-OCRv4 detection and recognition ONNX runtime
- `pillow` (v12.3.0) — image cropping and encoding
- `opencv-python` (v5.0.0.93) — computer vision contour, morphology, and circle detection
- `requests` (v2.34.2) — server-side multimodal API client
- `numpy` (v2.5.3) & `shapely` (v2.2.0) — geometric coordinate manipulations

### 4. Do I need to add an API key?
**No, an API key is NOT required.** ParseX works out-of-the-box using the local RapidOCR and computer vision fallback engine.
If you have a Gemini API key, you can optionally configure it to enable cloud vision analysis.

### 5. Exactly what environment variable do I need?
Add to `.env` or set in your environment:
```bash
VISION_API_KEY=your_gemini_api_key_here
```
Optional settings:
```bash
VISION_PROVIDER=gemini        # "gemini" or "none" (to force local fallback)
VISION_MODEL=gemini-2.5-flash # or gemini-1.5-flash
PORT=8000
HOST=127.0.0.1
```
*Note: The API key remains strictly server-side and is never exposed to browser code.*

### 6. How to run the application?
1. Start the ParseX server:
   ```bash
   python3 server.py
   ```
2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```
   Or open `Downloads/ParseX.html` directly in any web browser.

### 7. What test file/page should I use to demonstrate handwriting + figures?
Use `samples/test3_handwritten_avl.pdf` (or click the purple **"Handwritten AVL Sample"** button in the UI).
- It is a 10-page document where **Page 10** contains:
  - Handwritten C code: `struct Node* rotRight(struct Node* y){...}`
  - AVL tree diagram with node circles (`30`, `20`, `10`), transition arrow (`right rotate`), and balanced tree (`20` with `10` and `30`)
  - Rotation annotations: `LL case (left-left heavy)`
  - Algorithmic math notation: `bf(30) = h(L) - h(R) = 2 -> unbalanced`
  - Balance note: `after rotation all bf = 0, balanced`
- **In Ask tab, try asking:**
  - *"What rotation is shown in the diagram on page 10?"* → Answers: *"Right rotation (LL case)"* with Page 10 bounding box link!
  - *"What is the return statement of rotRight?"* → Answers: *"return x;"* with code line provenance!
  - *"What is the balance factor formula?"* → Answers: *"bf(30) = h(L) - h(R) = 2 -> unbalanced"*!

### 8. What limitations remain?
- Extremely low-resolution scans (< 75 DPI) or severe camera motion blur may require manual transcription verification (which ParseX faithfully flags).
- Overlapping handwritten annotations directly superimposed on complex photographic textures are flagged for human review.
- High-volume cloud vision requests are subject to Google Gemini API rate limits if `VISION_API_KEY` is configured.

---

## Running the Automated Test Suite

Run all 7 end-to-end pipeline tests:
```bash
python3 -m unittest tests/test_pipeline.py
```
Validates:
- `Test 1`: Digital PDF text & layout parsing
- `Test 2`: Scanned printed PDF OCR & bounding boxes
- `Test 3`: Page 10 Handwritten AVL Tree Rotation extraction & Q&A
- `Test 4`: Financial table multi-column alignment
- `Test 5`: Diagram and regional chart analytics
- `Test 6`: Mixed digital + visual composite page
- `Test 7`: Corrupted / invalid input handling
