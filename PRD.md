# Product Requirements Document (PRD)
## NeuralNumber — Multilingual Handwriting Notes App

**Version:** 1.0  
**Date:** 2026-04-13  
**Author:** Roshan  
**Status:** Active — In Development  

---

## 1. Executive Summary

NeuralNumber is a cross-platform intelligent notes application that converts hand-drawn strokes (via stylus or finger) into recognized multilingual text in real time. The system supports **Devanagari script**, **English (print)**, and **numeric digits** — 82 character classes in total.

The core intelligence runs as a **Python-based backend web service** (FastAPI or Flask), hosting a trained CNN/ResNet model for character recognition. Native mobile clients on **iOS (SwiftUI)** and **Android (Flutter/Kotlin)** communicate with this service via a REST/WebSocket API, providing the drawing canvas and note management UX. This architecture cleanly separates ML concerns (Python ecosystem) from the native device UI, enables faster iteration, and supports deploying model improvements without app store updates.

---

## 2. Problem Statement

Existing note-taking apps (Notability, GoodNotes) digitize ink but do not recognize or convert multilingual handwriting (specifically Devanagari + English mixed scripts) into searchable, editable text. Users who write in Hindi/Sanskrit alongside English have no seamless solution.

---

## 3. Goals and Non-Goals

### 3.1 Goals

| Priority | Goal |
|----------|------|
| P0 | Real-time handwritten character recognition (≥ 95% accuracy on isolated characters) |
| P0 | Support 82 classes: 10 digits, 26 English letters, 46 Devanagari characters |
| P0 | Python ML backend accessible via REST API (hosted locally or on cloud) |
| P1 | iOS native client (SwiftUI) using the Python backend |
| P1 | Android/cross-platform client (Flutter or Kotlin) using the same backend |
| P1 | Note management: create, tag, search, export (PDF/text) |
| P2 | User correction loop feeding back into model retraining pipeline |
| P2 | On-device fallback inference (CoreML for iOS, TFLite for Android) |
| P3 | Collaborative notes, LaTeX rendering, cloud sync |

### 3.2 Non-Goals

- Cursive English recognition (MVP targets print-style separated characters)
- OCR of pre-existing scanned documents
- Offline-first — MVP requires Python backend to be reachable

---

## 4. System Architecture

### 4.1 High-Level Architecture (v1 — Python Backend)

```
┌──────────────────────────────────────────────────────────────────┐
│                     Mobile Client Layer                          │
│                                                                  │
│   ┌──────────────────────┐      ┌──────────────────────────┐    │
│   │   iOS App (SwiftUI)  │      │  Android App (Flutter /  │    │
│   │   PencilKit Canvas   │      │  Kotlin) Canvas Input    │    │
│   │   Stroke → 28×28 img │      │  Stroke → 28×28 img      │    │
│   └──────────┬───────────┘      └────────────┬─────────────┘    │
│              │ HTTPS / WebSocket              │                  │
└──────────────┼────────────────────────────────┼──────────────────┘
               │                                │
               ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Python Backend (FastAPI)                           │
│                                                                 │
│  POST /recognize   ← base64 PNG (28×28 grayscale)              │
│  Returns: { char, script, confidence, top5 }                   │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Inference Engine                                         │  │
│  │  CNN / ResNet model (PyTorch)                             │  │
│  │  Preprocessing: normalize, invert, center-pad to 28×28   │  │
│  │  Script Router: class 0–9 → digit, 10–35 → EN, 36–81 → देव │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  POST /feedback    ← correction pairs for retraining queue     │
│  GET  /model/info  ← current model version, accuracy metrics   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Model Registry /    │
    │  Retraining Pipeline │
    │  (Python scripts)    │
    └──────────────────────┘
```

### 4.2 Architecture Decision Record (ADR): Python Backend vs. On-Device

| Criterion | Python Backend (chosen) | Pure On-Device (CoreML/TFLite) |
|-----------|------------------------|-------------------------------|
| ML iteration speed | ✅ Fast — redeploy server | ❌ Slow — rebuild & submit app |
| Cross-platform | ✅ One model serves iOS + Android | ❌ Separate CoreML + TFLite exports |
| Offline capability | ❌ Requires network | ✅ Fully offline |
| Inference latency | ~50–200ms (LAN), ~100–500ms (cloud) | <10ms |
| Model size on device | ✅ None | ❌ ~2–10 MB bundled |

**Decision:** Python backend for MVP development speed. On-device models (CoreML for iOS, TFLite for Android) are a P2 feature for offline fallback.

---

## 5. Python Backend Specification

### 5.1 Tech Stack

| Component  | Technology |
|------------|-----------|
| Web Framework | **FastAPI** (async, auto-docs, Pydantic validation) |
| ML Framework | **PyTorch** (model definition + inference) |
| Image Processing | **Pillow**, **OpenCV** (preprocessing pipeline) |
| Server | **Uvicorn** (ASGI) |
| Packaging | `pyproject.toml` / `requirements.txt` |
| Containerization | **Docker** (for cloud deployment) |
| Optional Deployment | **Railway / Render / AWS EC2** |

### 5.2 API Endpoints

#### `POST /recognize`

Accepts a single character image (28×28, grayscale) for classification.

**Request Body:**
```json
{
  "image_b64": "<base64-encoded PNG>",
  "session_id": "uuid-optional"
}
```

**Response:**
```json
{
  "character": "ग",
  "unicode": "U+0917",
  "script": "devanagari",
  "confidence": 0.983,
  "top5": [
    { "character": "ग", "confidence": 0.983 },
    { "character": "ग्", "confidence": 0.009 },
    ...
  ],
  "latency_ms": 12.4
}
```

**Script codes:** `"digit"` | `"english"` | `"devanagari"`

---

#### `POST /recognize/batch`

Accepts multiple character images (one note line) in a single request to minimize round-trips.

**Request Body:**
```json
{
  "images": ["<b64>", "<b64>", ...],
  "session_id": "uuid-optional"
}
```

**Response:** Array of individual `/recognize` responses.

---

#### `POST /feedback`

Submits a user correction for continuous learning.

```json
{
  "image_b64": "<base64 PNG>",
  "predicted": "ग",
  "corrected": "घ",
  "session_id": "uuid"
}
```

---

#### `GET /model/info`

Returns model metadata.

```json
{
  "model_name": "NeuralNumberCNN",
  "architecture": "ResNet-18-small",
  "version": "1.3.2",
  "num_classes": 82,
  "trained_on": ["EMNIST", "Devanagari-92k"],
  "val_accuracy": 0.9931
}
```

---

#### `GET /health`

Health check for deployment/monitoring.

```json
{ "status": "ok", "uptime_s": 3842 }
```

### 5.3 Preprocessing Pipeline (Server-Side)

```
Client sends raw stroke data → 28×28 PNG
         │
         ▼
  Load image (Pillow)
         │
  Convert to grayscale
         │
  Invert (white bg → black bg, white strokes → input tensor convention)
         │
  Normalize pixel values to [0, 1]
         │
  Add batch + channel dims: (1, 1, 28, 28)
         │
  torch.no_grad() → model(tensor)
         │
  Softmax → top-5 predictions
         │
  Map class index → character + script label
         │
  Return JSON response
```

### 5.4 Project Structure (Python Backend)

```
neuralnumber-backend/
├── main.py                  # FastAPI app, route definitions
├── model/
│   ├── architecture.py      # NeuralNumberCNN / ResNet class definitions
│   ├── inference.py         # InferenceEngine singleton, preprocessing
│   ├── class_map.json       # { 0: "0", ..., 36: "अ", ..., 81: "ज्ञ" }
│   └── weights/
│       └── neuralnumber_v1.3.pt
├── schemas/
│   ├── recognize.py         # Pydantic request/response models
│   └── feedback.py
├── utils/
│   ├── image_utils.py       # Base64 decode, preprocessing helpers
│   └── logging.py
├── feedback_store/          # Raw correction images + labels (for retraining)
├── Dockerfile
├── requirements.txt
└── tests/
    ├── test_api.py
    └── test_inference.py
```

---

## 6. Mobile Client Specifications

### 6.1 iOS Client (SwiftUI)

#### Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | **SwiftUI** |
| Drawing | **PencilKit** (`PKCanvasView`) |
| Networking | **URLSession** / **Combine** (async REST calls) |
| Local Storage | **SwiftData** |
| Cloud Sync | **CloudKit** (P3) |
| Language | **Swift 6** |

#### Core Features

1. **Ink Canvas** — Full-screen `PKCanvasView` with Apple Pencil pressure/tilt support and automatic palm rejection.
2. **Character Box Mode (MVP)** — Guided grid of character boxes (Option A from roadmap). User writes one character per box. Box boundary triggers recognition on lift.
3. **Real-Time Recognition** — On pen-lift, the client:
   - Crops the bounding box of the stroke.
   - Renders to a 28×28 grayscale PNG.
   - Sends `POST /recognize` to the Python backend.
   - Overlays the recognized character above the stroke within ~200ms.
4. **Script Badge** — Small indicator (🔢 digit / 🔤 EN / देव) next to each recognized character.
5. **Inline Correction** — Tap any recognized character to open a correction popover. Correction is sent via `POST /feedback`.
6. **Note Management** — Note list (title, timestamp, tag), note detail view with canvas + recognized text panel.
7. **Export** — PDF (ink + recognized text overlay) or plain `.txt`.

#### Networking Layer (Swift)

```swift
struct RecognizeRequest: Codable {
    let image_b64: String
    let session_id: String
}

struct RecognizeResponse: Codable {
    let character: String
    let script: String
    let confidence: Double
    let top5: [CharacterScore]
}

class NeuralNumberAPIClient {
    static let shared = NeuralNumberAPIClient()
    private let baseURL = URL(string: "http://localhost:8000")! // or cloud URL
    
    func recognize(imageData: Data) async throws -> RecognizeResponse {
        let b64 = imageData.base64EncodedString()
        let body = RecognizeRequest(image_b64: b64, session_id: sessionID)
        var request = URLRequest(url: baseURL.appendingPathComponent("/recognize"))
        request.httpMethod = "POST"
        request.httpBody = try JSONEncoder().encode(body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(RecognizeResponse.self, from: data)
    }
}
```

#### iOS Project Structure

```
NeuralNumberApp/
├── App/
│   ├── NeuralNumberApp.swift
│   └── ContentView.swift
├── Features/
│   ├── Canvas/
│   │   ├── CanvasView.swift          # PencilKit wrapper
│   │   ├── CharacterBoxGrid.swift    # Guided character box layout
│   │   ├── StrokeRenderer.swift      # Stroke → 28×28 PNG
│   │   └── CanvasViewModel.swift     # Recognition orchestration
│   ├── Recognition/
│   │   ├── APIClient.swift           # REST calls to Python backend
│   │   ├── RecognitionOverlay.swift  # Character label overlay
│   │   └── CorrectionPopover.swift   # Inline correction UI
│   ├── Notes/
│   │   ├── Note.swift                # SwiftData model
│   │   ├── NoteListView.swift
│   │   └── NoteDetailView.swift
│   └── Settings/
│       └── SettingsView.swift        # Backend URL, model info, language mode
├── Resources/
│   └── Assets.xcassets
└── Tests/
    ├── APIClientTests.swift
    └── StrokeRendererTests.swift
```

---

### 6.2 Android / Cross-Platform Client (Flutter)

#### Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | **Flutter** (Dart) — or native Kotlin/Jetpack Compose |
| Drawing | **CustomPainter** + `GestureDetector` (Flutter) |
| Networking | **`http`** / **`dio`** package |
| Local Storage | **SQLite** via `sqflite` |
| Language | **Dart** (Flutter) |

> **Recommendation:** Use Flutter for maximum code reuse between iOS and Android if SwiftUI is not strictly required. Use Kotlin/Jetpack Compose if you want tighter Android system integration (S-Pen support on Samsung devices).

#### Flutter Drawing Canvas

```dart
class CharacterCanvas extends StatefulWidget { ... }

class _CharacterCanvasState extends State<CharacterCanvas> {
  List<Offset?> _points = [];

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onPanUpdate: (details) {
        setState(() => _points.add(details.localPosition));
      },
      onPanEnd: (_) {
        setState(() => _points.add(null));
        _sendForRecognition();
      },
      child: CustomPaint(
        painter: StrokePainter(points: _points),
        size: Size(280, 280),
      ),
    );
  }

  Future<void> _sendForRecognition() async {
    final imageBytes = await _renderToBytes();
    final response = await NeuralNumberAPI.recognize(imageBytes);
    setState(() { _recognized = response.character; });
  }
}
```

#### Flutter API Client

```dart
class NeuralNumberAPI {
  static const _baseUrl = 'http://localhost:8000';

  static Future<RecognizeResponse> recognize(Uint8List imageBytes) async {
    final b64 = base64Encode(imageBytes);
    final res = await http.post(
      Uri.parse('$_baseUrl/recognize'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'image_b64': b64}),
    );
    return RecognizeResponse.fromJson(jsonDecode(res.body));
  }
}
```

---

## 7. ML Model Specification

### 7.1 Architecture Roadmap

| Phase | Architecture | Classes | Target Accuracy | Deployment |
|-------|-------------|---------|-----------------|------------|
| MVP | CNN (2×Conv blocks) | 82 | ≥ 95% | Python backend |
| v1.1 | ResNet-18 (small) | 82 | ≥ 99% | Python backend |
| v2.0 | CRNN + CTC | Full-line | Sequence WER < 5% | Python backend |
| P2 | MobileNetV2 | 82 | ≥ 98% | CoreML (iOS) + TFLite (Android) |

### 7.2 Training Datasets

| Dataset | Script | Samples | Source |
|---------|--------|---------|--------|
| EMNIST (ByClass) | Digits + English | ~814,000 | NIST |
| Devanagari Character Dataset | Devanagari | ~92,000 | Kaggle |
| **Total** | Mixed | **~906,000** | Combined |

### 7.3 Class Map (82 classes)

```
0–9    → '0' through '9'      (digits)
10–35  → 'A' through 'Z'      (English uppercase)
36–81  → 'अ','आ',...,'ज्ञ'    (46 Devanagari characters)
```

### 7.4 Model Serving Architecture

```python
# inference.py — singleton loaded at FastAPI startup

import torch
from torchvision import transforms
from PIL import Image
import io, base64, json

class InferenceEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        with open("model/class_map.json") as f:
            self.class_map = json.load(f)
        self.model = torch.load("model/weights/neuralnumber_v1.pt", map_location="cpu")
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, image_b64: str) -> dict:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("L")
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
        top5 = torch.topk(probs, 5)
        return {
            "character": self.class_map[str(top5.indices[0].item())],
            "confidence": round(top5.values[0].item(), 4),
            "top5": [
                {"character": self.class_map[str(i.item())], "confidence": round(v.item(), 4)}
                for i, v in zip(top5.indices, top5.values)
            ]
        }
```

---

## 8. Stroke Segmentation Strategy

The client is responsible for segmenting individual characters before sending to the backend.

| Option | Description | MVP? |
|--------|-------------|------|
| **A — Character Box Grid** | Fixed grid of boxes; user writes one character per box. Box boundary triggers recognition on pen-lift. | ✅ MVP |
| **B — Connected Component Analysis** | Render all strokes to a bitmap; find connected components as character candidates. | v1.1 |
| **C — CRNN + CTC (server-side)** | Send full-line image to backend; backend returns a character sequence. No client-side segmentation. | v2.0 |

**MVP uses Option A.** The client renders the 28×28 image from the box content and sends it to `/recognize`.

---

## 9. Data Flow: End-to-End (MVP)

```
1. User opens NeuralNumber app (iOS or Android)
2. Creates a new note → canvas opens in character-box mode
3. User writes a character in a box with Apple Pencil / stylus
4. On pen-lift:
     a. Client crops box → renders 28×28 grayscale PNG
     b. Client encodes to base64
     c. POST /recognize → Python backend
     d. Backend preprocesses → CNN inference → returns { char, confidence }
     e. Client overlays recognized character above the stroke
5. User sees recognized character within ~100–300ms
6. User reviews; taps to correct if wrong → POST /feedback
7. Notes saved locally (SwiftData / SQLite)
8. User exports note as PDF or text
```

---

## 10. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| **Latency** | Recognition response < 300ms on LAN; < 600ms on cloud (4G) |
| **Accuracy** | ≥ 95% top-1 accuracy on held-out test set (balanced across 82 classes) |
| **Throughput** | Backend handles ≥ 20 concurrent recognition requests |
| **Availability** | 99.5% uptime target (for hosted backend) |
| **Security** | HTTPS for all API calls; session token auth (P1) |
| **Offline fallback** | On-device CoreML (iOS) / TFLite (Android) as P2 feature |
| **Privacy** | User strokes and corrections stored locally; only sent to backend with consent |

---

## 11. Phase Roadmap (Updated)

### Phase 1 — Python Backend + Model Training *(Current)*
- [ ] Train CNN on EMNIST + Devanagari dataset (82 classes)
- [ ] Build FastAPI backend with `/recognize`, `/feedback`, `/health`
- [ ] Containerize with Docker
- [ ] Validate API with integration tests (Pytest + httpx)
- [ ] Deploy locally (uvicorn) for client development

### Phase 2 — iOS Client (SwiftUI)
- [ ] PencilKit canvas with character-box grid (Option A)
- [ ] Stroke renderer (box → 28×28 PNG)
- [ ] REST client connecting to Python backend
- [ ] Character overlay + script badge UI
- [ ] Inline correction with feedback API call
- [ ] Note management (SwiftData)
- [ ] Export (PDF / text)

### Phase 3 — Android / Flutter Client
- [ ] Flutter canvas with CustomPainter stroke input
- [ ] Same REST client → Python backend
- [ ] Feature parity with iOS client (MVP features)
- [ ] Stylus support (Samsung S-Pen via `MotionEvent` detection)

### Phase 4 — Model Upgrade (ResNet + CRNN)
- [ ] Upgrade backend model to ResNet-18-small (target: 99%+)
- [ ] Implement CRNN + CTC for full-line recognition (Option C)
- [ ] Add `/recognize/line` endpoint for line-image input
- [ ] Update mobile clients to support line-mode

### Phase 5 — On-Device Fallback (P2)
- [ ] Export PyTorch model → CoreML (iOS offline mode)
- [ ] Export PyTorch model → TFLite (Android offline mode)
- [ ] Graceful degradation: if backend unreachable, switch to on-device
- [ ] On-device fine-tuning via CoreML `MLUpdateTask` (iOS)

### Phase 6 — Production Hardening (P3)
- [ ] Model registry with versioned weights
- [ ] Retraining pipeline from collected feedback
- [ ] CloudKit sync (iOS) / Firebase sync (Android)
- [ ] LaTeX math symbol recognition
- [ ] A/B testing new architectures via backend feature flags

---

## 12. Architecture Comparison — On-Device vs. Backend

```
                ┌─────────────────────────────────────────────┐
                │         Architecture Decision Matrix         │
                ├──────────────┬──────────────┬───────────────┤
                │ Criterion    │ Backend (v1) │ On-Device (P2) │
                ├──────────────┼──────────────┼───────────────┤
                │ Dev speed    │ ✅ Fast       │ ❌ Slow        │
                │ Cross-plat.  │ ✅ One model  │ ❌ 2 exports   │
                │ Offline      │ ❌ Network req│ ✅ Always works │
                │ Latency      │ ⚠ 100-600ms  │ ✅ <10ms       │
                │ Model update │ ✅ Instant    │ ❌ App update  │
                │ Device space │ ✅ None       │ ❌ 2-10 MB    │
                └──────────────┴──────────────┴───────────────┘
```

---

## 13. Alternative Architecture Consideration — Swift/Flutter Wrapping a Python Web App

An alternative to a native client calling a REST API is embedding a **local Python web server** in the native app wrapper:

```
Option A (current plan):
  Native UI ──HTTP──▶ Remote Python backend (cloud/LAN)

Option B (embedded server):
  Native UI ──WebView/Local HTTP──▶ Python Flask/FastAPI running in-process
  (using Kivy, BeeWare, or a Python.framework embed in Swift)

Option C (future PWA):
  Python FastAPI serves both the API AND a web frontend (HTML/JS)
  Swift/Flutter wraps the web app in a WKWebView / Flutter WebView
```

**For MVP:** Use Option A (native client + remote API). This is standard production architecture.  
**Option B** (embedded Python) is experimental and has significant packaging friction on iOS App Store.  
**Option C** (PWA wrapping) is viable for a demo/prototype but sacrifices native drawing performance.

---

## 14. Key Resources

| Resource | URL |
|----------|-----|
| FastAPI Docs | https://fastapi.tiangolo.com |
| PyTorch Model Serving | https://pytorch.org/serve/ |
| EMNIST Dataset | https://arxiv.org/abs/1702.05373 |
| Devanagari Dataset (Kaggle) | https://www.kaggle.com/datasets/ashokpant/devanagari-character-dataset-large |
| CoreML Tools | https://coremltools.readme.io |
| PencilKit Docs | https://developer.apple.com/documentation/pencilkit |
| Flutter CustomPainter | https://api.flutter.dev/flutter/rendering/CustomPainter-class.html |
| TFLite for Flutter | https://pub.dev/packages/tflite_flutter |
| CTC Loss Explained | https://distill.pub/2017/ctc/ |
| MobileNetV2 Paper | https://arxiv.org/abs/1801.04381 |
| Vision Transformer (ViT) | https://arxiv.org/abs/2010.11929 |

---

## Appendix A — Devanagari Character Classes (36–81)

**Vowels (स्वर):** अ आ इ ई उ ऊ ऋ ए ऐ ओ  
**Consonants (व्यंजन):** क ख ग घ ङ | च छ ज झ ञ | ट ठ ड ढ ण | त थ द ध न | प फ ब भ म | य र ल व | श ष स ह | क्ष त्र ज्ञ

> [!WARNING]
> Conjunct characters (क्ष, त्र, ज्ञ) are visually complex. Treat them as distinct classes but evaluate accuracy separately. If accuracy < 80% on conjuncts, consider deferring to v1.1.

---

## Appendix B — MVP Acceptance Criteria

| Feature | Criterion | Pass |
|---------|-----------|------|
| Backend `/recognize` | Returns correct class for 95%+ of EMNIST test images | - |
| Backend latency | p95 < 200ms on local network | - |
| iOS canvas | Stroke renders correctly in character box | - |
| iOS recognition | Recognized character overlaid within 300ms of pen-lift | - |
| iOS correction | Correction popover submits feedback and updates display | - |
| Note export | PDF export includes both ink and text layers | - |
| Flutter canvas | Stroke input works with mouse and touch | - |
| Flutter recognition | Same API integration, character overlaid correctly | - |
