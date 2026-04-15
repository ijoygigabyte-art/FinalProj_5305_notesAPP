## Phase 3 — Real-Time iPad + Apple Pencil Notes App

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                   iPad App (Swift / SwiftUI)         │
│                                                     │
│  ┌──────────────┐    ┌────────────────────────────┐ │
│  │  PencilKit   │───▶│   Stroke Segmentation      │ │
│  │  Canvas      │    │   (character isolation)     │ │
│  └──────────────┘    └────────────┬───────────────┘ │
│                                   │                  │
│                      ┌────────────▼───────────────┐ │
│                      │   CoreML Model (.mlmodel)  │ │
│                      │   CNN inference on-device   │ │
│                      └────────────┬───────────────┘ │
│                                   │                  │
│                      ┌────────────▼───────────────┐ │
│                      │   Text Reconstruction      │ │
│                      │   + Note Storage (SwiftData)│ │
│                      └────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 3.2 Model Export Pipeline (Python → CoreML)

The scratch NumPy model cannot be directly converted to CoreML. You must:

1. **Re-implement the trained architecture in PyTorch** — copy weights from your NumPy arrays into `torch.nn` layers.
    ```python
    import torch, torch.nn as nn, coremltools as ct

    class NeuralNumberCNN(nn.Module):
        def __init__(self, num_classes=82):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(0.25),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(0.25),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    # Load your NumPy weights into this model
    # Convert
    model_pt = NeuralNumberCNN()
    model_pt.eval()
    example = torch.randn(1, 1, 28, 28)
    traced = torch.jit.trace(model_pt, example)
    mlmodel = ct.convert(traced, inputs=[ct.ImageType(shape=(1, 1, 28, 28))])
    mlmodel.save("NeuralNumber.mlmodel")
    ```

2. **Validate CoreML output** matches NumPy output on the same test images before deploying.

### 3.3 iPad App Implementation

#### Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | **SwiftUI** (declarative, modern) |
| Drawing | **PencilKit** (`PKCanvasView`) — native Apple Pencil support with pressure, tilt, azimuth |
| ML Inference | **CoreML** + **Vision** framework |
| Storage | **SwiftData** (local) + **CloudKit** (cross-device sync) |
| Language | **Swift 6** |

#### Core Features

1. **Ink Canvas** — Full-screen `PKCanvasView` with configurable pen thickness, color, and eraser. Support palm rejection automatically via PencilKit.
2. **Real-Time Recognition** — As the user pauses writing (debounce ~0.5s after last stroke), automatically:
   - Segment individual characters from the stroke data.
   - Render each character to a 28×28 grayscale image.
   - Run CoreML inference.
   - Overlay recognized text above the handwriting.
3. **Script Detection** — Use the top-1 prediction's class range to determine the script:
   - Classes 0–9 → Digit
   - Classes 10–35 → English letter
   - Classes 36–81 → Devanagari character
4. **Editable Text Layer** — Tapping recognized text opens an inline editor to correct misrecognitions. Corrections are logged and can later be used for fine-tuning.
5. **Note Management** — Create, rename, tag, search, and delete notes. Export to PDF or plain text.

#### Stroke Segmentation Strategy

This is the **hardest unsolved problem** in the pipeline. Isolated character recognition (what the CNN does) is different from connected/cursive segmentation. Recommended approach:

```
Option A:  Force print-style input (one stroke per character box)
           → Simplest. Grid-based. High accuracy.

Option B:  Connected Component Analysis on the rendered bitmap
           → Works for printed Devanagari and block English.
           → Fails on cursive.

Option C:  Use an RNN/Transformer sequence model (see §4) to go from
           stroke sequences → text directly, bypassing segmentation.
           → Best accuracy. Most complex.
```

For an MVP, start with **Option A** (guided character boxes) and iterate toward Option B.

### 3.4 Xcode Project Structure

```
NeuralNumberApp/
├── App/
│   ├── NeuralNumberApp.swift          # @main entry point
│   └── ContentView.swift              # Tab-based root view
├── Features/
│   ├── Canvas/
│   │   ├── CanvasView.swift           # PencilKit wrapper
│   │   ├── StrokeSegmenter.swift      # Character isolation logic
│   │   └── CanvasViewModel.swift      # Debounce + recognition orchestration
│   ├── Recognition/
│   │   ├── CharacterClassifier.swift  # CoreML inference wrapper
│   │   ├── ScriptDetector.swift       # Digit / English / Devanagari router
│   │   └── NeuralNumber.mlmodel       # Exported model
│   ├── Notes/
│   │   ├── Note.swift                 # SwiftData model
│   │   ├── NoteListView.swift         # All notes browser
│   │   └── NoteDetailView.swift       # Single note with canvas
│   └── Settings/
│       └── SettingsView.swift         # Model version, language toggle, etc.
├── Resources/
│   └── Assets.xcassets
└── Tests/
    ├── ClassifierTests.swift
    └── SegmenterTests.swift
```

---

## Phase 4 — Alternative & Advanced Architectures to Consider

Beyond the baseline CNN, these architectures are worth implementing or benchmarking. Each trades off complexity vs. accuracy for handwriting recognition tasks.

### 4.1 ResNet (Residual Networks)

**Why it matters:** Skip connections solve the vanishing gradient problem in deep networks. A ResNet-18 adapted for 28×28 grayscale images can reach **99.7%+** on MNIST.

**Key idea:** `output = F(x) + x` (identity shortcut). If the layer can't learn anything useful, it just passes the input through unchanged.

```
ResidualBlock:
    x ──┬── Conv → BN → ReLU → Conv → BN ──[+]── ReLU → out
        │                                    ▲
        └────────────────────────────────────┘  (identity shortcut)
```

**Implementation difficulty:** Moderate. Add a `ResidualBlock` class that wraps two Conv2D + BN layers and adds the input. Handle dimension mismatches with a 1×1 projection convolution.

**Recommended config for 28×28:**
```
Conv(1→16, 3×3)  →  2× ResBlock(16)  →  2× ResBlock(32, stride=2)  →  AvgPool  →  FC(82)
```

---

### 4.2 MobileNetV2 / Depthwise Separable Convolutions

**Why it matters:** If the model will run on-device (iPad), inference speed is critical. MobileNet replaces standard convolutions with depthwise separable convolutions, reducing compute by **8–9×** with minimal accuracy loss.

**Key idea:**
```
Standard Conv:  C_in × C_out × K × K  multiplications per pixel
Depthwise Sep:  C_in × K × K  +  C_in × C_out   (much fewer)
```

**Implementation difficulty:** Moderate. Implement `DepthwiseConv2D` (one filter per input channel) and `PointwiseConv2D` (1×1 conv to mix channels).

**When to use:** When you need real-time inference (< 10ms per character) on an A-series or M-series chip via CoreML. ANE (Apple Neural Engine) handles depthwise convolutions extremely efficiently.

---

### 4.3 Vision Transformer (ViT)

**Why it matters:** Transformers have matched or exceeded CNNs on image classification. A small ViT adapted for 28×28 inputs is an excellent learning exercise and may outperform CNNs on multi-script recognition where global context matters (e.g., distinguishing similar-looking Devanagari and English characters).

**Key idea:** Split the image into patches → linear embed each patch → feed the sequence to a standard Transformer encoder → classify from the `[CLS]` token.

```
Image (28×28) → 49 patches of 4×4 → Linear(16 → 64) → + position embeddings
    → L× [MultiHeadSelfAttention → LayerNorm → FFN → LayerNorm]
    → CLS token → Linear(64 → 82) → Softmax
```

**Recommended config:**
| Param | Value |
|-------|-------|
| Patch size | 4×4 |
| Embedding dim | 64 |
| Heads | 4 |
| Transformer layers | 4 |
| FFN hidden dim | 128 |

**Implementation difficulty:** High. Requires implementing multi-head self-attention, positional embeddings, and layer normalization from scratch. However, you already have the linear algebra foundation from your MLP.

**Caveat:** ViT needs more data than CNNs to generalize well. With only 92k Devanagari samples, consider data augmentation or pre-training on a larger combined dataset first.

---

### 4.4 CRNN (CNN + RNN) for Sequence Recognition

**Why it matters:** This is the architecture that solves the **stroke segmentation problem** from Phase 3. Instead of classifying isolated characters, a CRNN reads an entire line of handwriting and outputs a sequence of characters — no explicit segmentation needed.

**Architecture:**
```
Input image (1, 32, W)       ← variable-width line image
    → CNN feature extractor  → (512, 1, W')   feature columns
    → Reshape to (W', 512)   → sequence of feature vectors
    → Bidirectional LSTM (2 layers, 256 hidden)
    → Linear(512 → num_classes + 1)    ← +1 for CTC blank token
    → CTC Decode
```

**Key component — CTC Loss (Connectionist Temporal Classification):**
- Allows the model to predict a sequence of characters from a sequence of feature vectors without knowing the alignment between them.
- The model outputs a character (or "blank") at every time step; CTC collapses repeated characters and removes blanks to produce the final string.

**Implementation difficulty:** Very High. The CTC forward-backward algorithm is non-trivial to implement from scratch. Consider using a framework (PyTorch `nn.CTCLoss`) for this component. The CNN feature extractor can still be your scratch implementation.

**When to use:** Phase 3, Option C — when you want to skip character segmentation entirely for the iPad app.

---

### 4.5 Capsule Networks (CapsNet)

**Why it matters:** CapsNet preserves spatial hierarchies that CNNs discard via max-pooling. Each "capsule" outputs a vector (not a scalar), encoding both the probability of a feature's existence and its instantiation parameters (position, rotation, size). This is powerful for handwriting where the same character can appear in many poses.

**Architecture (Hinton's original for MNIST):**
```
Conv(1→256, 9×9) → ReLU
    → PrimaryCaps: Conv(256→32×8, 9×9, stride=2) → squash
    → DigitCaps: Dynamic routing (3 iterations) → 82 capsules × 16D
    → Length of each capsule vector = class probability
```

**Implementation difficulty:** Very High. Dynamic routing between capsules is the core novelty and is mathematically involved. Reconstruction regularization (decoder sub-network) adds another training signal.

**When to use:** Research or exploration. CapsNets haven't seen widespread production use but are academically interesting, especially for understanding equivariance in recognition.

---

### 4.6 Architecture Comparison Summary

| Architecture | MNIST Accuracy | Params (approx.) | Inference Speed (iPad) | Implementation Effort | Best For |
|---|---|---|---|---|---|
| **MLP** (current) | 98.71% | ~260 K | < 1 ms | ✅ Done | Baseline |
| **CNN** (Phase 1) | 99.3–99.5% | ~600 K | ~2 ms | Medium | Default upgrade |
| **ResNet-18 (small)** | 99.5–99.7% | ~1 M | ~3 ms | Medium | Higher accuracy |
| **MobileNetV2** | 99.2–99.4% | ~300 K | < 1 ms | Medium | On-device speed |
| **ViT (tiny)** | 99.0–99.4% | ~500 K | ~5 ms | High | Multi-script context |
| **CRNN + CTC** | N/A (sequence) | ~3 M | ~15 ms/line | Very High | Full-line recognition |
| **CapsNet** | 99.6% | ~8 M | ~10 ms | Very High | Research / equivariance |

**Recommendation:** Implement **CNN (Phase 1)** → **ResNet (quick upgrade)** → **CRNN + CTC (for Phase 3 line recognition)**. Skip CapsNet unless it's for academic exploration. Use MobileNetV2 when optimizing the CoreML export for iPad.

---

## Phase 5 — Production Hardening & Stretch Goals

### 5.1 Model Improvements
- [ ] **Knowledge Distillation** — Train a small MobileNet "student" from a large ResNet "teacher" for deployment.
- [ ] **Quantization** — Use CoreML's `quantize_weights` to convert Float32 → Int8, reducing model size by ~4× with < 0.5% accuracy loss.
- [ ] **On-Device Fine-Tuning** — Use CoreML's `MLUpdateTask` to let users improve the model with their own handwriting corrections directly on the iPad.

### 5.2 App Features
- [ ] **Multi-language keyboard toggle** — Switch between Devanagari / English recognition modes, or let the model auto-detect.
- [ ] **LaTeX rendering** — Recognize mathematical symbols and render equations.
- [ ] **Collaboration** — Real-time shared notes via CloudKit or WebSockets.
- [ ] **Apple Watch companion** — Quick note capture with scribble input on watchOS.
- [ ] **Shortcuts / Siri integration** — "Hey Siri, create a new NeuralNumber note."

### 5.3 Backend / MLOps
- [ ] **Model Registry** — Version models; A/B test new architectures against production.
- [ ] **Feedback Loop** — Collect user corrections → retrain pipeline → push updated CoreML model via OTA (over-the-air updates using CloudKit or Firebase).
- [ ] **Monitoring** — Track per-class accuracy drift in production to detect when the model degrades on specific scripts.

---

## Appendix A — Devanagari Character Reference

The 46 Devanagari characters to support:

**Vowels (स्वर):**
| अ | आ | इ | ई | उ | ऊ | ऋ | ए | ऐ | ओ |
|---|---|---|---|---|---|---|---|---|---|

**Consonants (व्यंजन):**
| क | ख | ग | घ | ङ |
|---|---|---|---|---|
| च | छ | ज | झ | ञ |
| ट | ठ | ड | ढ | ण |
| त | थ | द | ध | न |
| प | फ | ब | भ | म |
| य | र | ल | व | - |
| श | ष | स | ह | - |
| क्ष | त्र | ज्ञ | - | - |

> [!WARNING]
> Conjunct characters (संयुक्त अक्षर) like क्ष, त्र, ज्ञ are visually complex. They may need to be treated as separate classes or excluded from the initial model until accuracy on base characters is solid.

## Appendix B — Key Resources

| Resource | URL |
|----------|-----|
| EMNIST Paper | https://arxiv.org/abs/1702.05373 |
| Devanagari Dataset (Kaggle) | https://www.kaggle.com/datasets/ashokpant/devanagari-character-dataset-large |
| CoreML Tools Docs | https://coremltools.readme.io |
| PencilKit Documentation | https://developer.apple.com/documentation/pencilkit |
| CTC Loss Explained | https://distill.pub/2017/ctc/ |
| Im2Col Tutorial | https://cs231n.github.io/convolutional-networks/#conv |
| ResNet Paper | https://arxiv.org/abs/1512.03385 |
| MobileNetV2 Paper | https://arxiv.org/abs/1801.04381 |
| Vision Transformer (ViT) | https://arxiv.org/abs/2010.11929 |
| Capsule Networks Paper | https://arxiv.org/abs/1710.09829 |