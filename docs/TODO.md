# Book Spine Detector Roadmap

Last updated: 2026-08-31

This document tracks the work required to move from the current public-dataset prototype to an
offline, real-time Android and iOS application for factory book-count QC.

## Current status

- [x] Create a Python training, evaluation, inference, and ONNX-export scaffold.
- [x] Train the LR-ASPP MobileNetV3-Large separator model on 1,249 training images.
- [x] Save resumable `best.pt` and `last.pt` checkpoints to Google Drive.
- [x] Stop training at epoch 35 after eight epochs without validation-loss improvement.
- [x] Evaluate `last.pt` on the public test set: Dice 0.646, IoU 0.485, count MAE 1.89,
  exact-count accuracy 24.2%.
- [x] Confirm at least one clean, single-stack image is counted correctly.
- [x] Export support is implemented for ONNX opset 17.
- [ ] Record and validate the final ONNX artifact produced from the selected checkpoint.
- [ ] Establish performance on manually counted factory images.

The current public metrics are feasibility results, not manufacturing acceptance results. The
public data contains library shelves, multiple groups, mixed orientations, and noisy annotations.
The product scenario should instead frame and count one factory stack at a time.

## Product definition and acceptance gates

- [ ] Write the exact QC workflow: who scans, when they scan, expected count range, and what
  happens when the result is uncertain.
- [ ] Decide whether one camera frame may contain only one stack or whether automatic stack ROI
  detection is required.
- [ ] Agree on the production exact-count target with QC stakeholders. Do not use Dice as the
  release metric.
- [ ] Agree on acceptable undercount versus overcount risk and whether their costs differ.
- [ ] Define a low-confidence path that asks for another scan or sends the item to human QC.
- [ ] Define supported minimum Android/iOS versions and representative low-, mid-, and high-end
  devices.
- [ ] Define initial mobile budgets: model size, application size, memory, battery, thermal load,
  p95 inference latency, and minimum count-update rate.
- [ ] Require fully offline inference unless the factory explicitly approves network processing.

Suggested milestone gates, to be confirmed by QC:

| Gate | Dataset | Purpose | Exit criterion |
|---|---|---|---|
| Feasibility | Public images | Prove the pipeline | Completed; not a quality gate |
| Factory baseline | 50–100 factory photos | Measure domain gap | Report exact accuracy and MAE |
| Fine-tuning pilot | 500–1,500 labeled photos | Improve model | Meet agreed pilot threshold |
| Locked factory test | 200+ held-out photos | Release decision | Meet production threshold |
| Shadow operation | Live line, human still authoritative | Operational validation | Stable results across shifts |

## Phase 1 — Factory data and ground truth

### Capture protocol

- [ ] Create a short photo-capture guide with good and bad examples.
- [ ] Keep one full stack inside a visible framing guide; do not crop the first or last book.
- [ ] Capture horizontal and vertical stacks if both are valid factory cases.
- [ ] Include the full supported count range and book-thickness range.
- [ ] Cover multiple titles, cover materials, colors, reflective laminates, and binding styles.
- [ ] Cover daylight, factory lighting, shadows, glare, blur, and low contrast.
- [ ] Cover permitted camera angles, distances, rotations, and background clutter.
- [ ] Include different phones and camera sensors.
- [ ] Capture multiple production days, lines, shifts, and operators.
- [ ] Avoid collecting many adjacent video frames as if they were independent examples.

### Labels and storage

- [ ] Record a human-verified integer count for every image.
- [ ] Record stack ROI, orientation, acquisition device, production line, shift, and batch where
  permitted.
- [ ] Annotate book boundaries or book polygons for a representative subset used for fine-tuning.
- [ ] Add an `uncountable`/`needs_review` label for severe occlusion or an incomplete stack.
- [ ] Write an annotation guide for partial books, gaps, straps, wrapping, and damaged books.
- [ ] Double-review the locked test set and resolve count disagreements.
- [ ] Store factory images outside Git; version manifests and annotation revisions.
- [ ] Document retention, access, privacy, and factory-data ownership rules.

### Split strategy

- [ ] Split by physical stack, production batch, or capture session—not randomly by image.
- [ ] Prevent near-duplicate frames from appearing in both training and evaluation sets.
- [ ] Keep a locked factory test set that is never used for threshold or model selection.
- [ ] Maintain smaller `train`, `validation`, and `test` manifests with stable image IDs.

## Phase 2 — Measurement and model improvement

### Establish the factory baseline

- [ ] Run the current `best.pt` model on 50–100 unseen, manually counted factory images.
- [ ] Save filename, actual count, predicted count, signed error, absolute error, confidence,
  orientation, latency, and model version to CSV/JSON.
- [ ] Save overlays for every wrong result and a sample of correct results.
- [ ] Report exact-count accuracy, count MAE, undercount rate, overcount rate, and error by true
  count bucket.
- [ ] Separate clean single-stack results from invalid/multi-stack inputs.
- [ ] Create an error taxonomy: missed boundary, duplicate boundary, wrong orientation, glare,
  crop, occlusion, blur, background line, mixed stacks, or annotation error.

### Post-processing

- [ ] Add automated parameter search using validation data only.
- [ ] Tune pixel threshold, peak height, peak prominence, smoothing window, and minimum boundary
  distance as one global configuration—not per image.
- [ ] Make exact-count accuracy the primary model-selection metric; use MAE and segmentation
  metrics as secondary diagnostics.
- [ ] Save `best_count.pt` or a model manifest whenever validation exact-count accuracy improves.
- [ ] Save per-epoch checkpoints temporarily so count-based selection is possible after training.
- [ ] Detect and reject ambiguous orientation instead of silently choosing the stronger profile.
- [ ] Merge duplicate edges from adjacent book annotations more robustly.
- [ ] Explore adaptive minimum distance based on image scale and estimated book thickness.
- [ ] Calibrate confidence against actual exact-count correctness.
- [ ] Define a confidence threshold below which the app asks the operator to rescan.

### Training and architecture experiments

- [ ] Fine-tune the public pretrained model on factory separator annotations.
- [ ] Compare public-only, factory-only, and public-then-factory fine-tuning.
- [ ] Add augmentations that match the factory: exposure, glare, blur, perspective, crop, sensor
  noise, wrapping, and mild rotation.
- [ ] Remove or repair high-impact public annotation outliers.
- [ ] Test 512, 640, and higher input resolutions on thin books.
- [ ] Revisit separator width and positive-class weighting.
- [ ] Compare boundary segmentation with direct book-instance segmentation and oriented detection.
- [ ] Consider a two-stage pipeline: stack ROI detection followed by separator counting.
- [ ] Consider direct count regression only as an auxiliary signal; retain visual boundaries for
  auditability.
- [ ] Track every experiment with immutable config, dataset revision, seed, metrics, and artifact
  checksum.
- [ ] Stop an experiment when the agreed validation metric plateaus; do not extend epochs only to
  reach a round number.

## Phase 3 — Lock the portable model artifact

- [ ] Select a checkpoint using factory validation exact-count accuracy.
- [ ] Export a fixed-input ONNX model and record opset, input/output names, normalization, color
  order, resize/letterbox behavior, and output semantics.
- [ ] Create `models/exported/model_manifest.json` containing model version, SHA-256, input size,
  thresholds, class mapping, training dataset revision, and metrics.
- [ ] Validate the ONNX graph with `onnx.checker`.
- [ ] Compare PyTorch and ONNX Runtime logits/probabilities on a fixed golden set with numeric
  tolerances.
- [ ] Compare final counts and boundary coordinates between Python and ONNX Runtime.
- [ ] Benchmark FP32 first; then test FP16 and INT8 quantization independently.
- [ ] Reject quantization if factory exact-count accuracy drops beyond the agreed tolerance.
- [ ] Evaluate ONNX versus ORT format and a reduced-operator custom runtime for application size.
- [ ] Define model rollback and version-compatibility rules.
- [ ] Keep sensitive golden images outside Git; commit only approved fixtures or synthetic cases.

ONNX Runtime provides supported mobile packages for Android (`onnxruntime-android`) and iOS
(`onnxruntime-objc`/`onnxruntime-c`). It also supports reduced custom builds after the model's
operator set is known: <https://onnxruntime.ai/docs/tutorials/mobile/>.

## Phase 4 — Shared C++ core and platform bindings

### Production architecture

Use native platform camera APIs and keep the complete inference pipeline in one shared C++ library.
The C++ core owns preprocessing, ONNX Runtime inference, post-processing, count stabilization,
confidence policy, and shared domain types. Android calls it through a narrow JNI layer; iOS calls
it through an Objective-C++ wrapper with a small Objective-C/C-compatible public surface. This
keeps model behavior identical while preserving native Kotlin and Swift user interfaces.

```text
Android CameraX                         iOS AVFoundation
       |                                      |
   Kotlin UI                               Swift UI
       | JNI                         Objective-C++ bridge
       +------------------+-------------------+
                          |
                 C++ book-counter-core
          preprocess / ONNX Runtime / peaks
          count / confidence / stabilization
```

ONNX Runtime officially supports C/C++ on both Android and iOS:
<https://onnxruntime.ai/docs/tutorials/mobile/>.

### C++ core tasks

- [ ] Create `core/CMakeLists.txt` with `include/`, `src/`, `tests/`, and platform binding folders.
- [ ] Define a narrow, versioned public API that does not expose ONNX Runtime objects to apps.
- [ ] Port letterbox metadata and coordinate transforms from Python.
- [ ] Port probability projection, smoothing, peak detection, orientation selection, count, and
  confidence calculations.
- [ ] Define stable C++ value types: `ModelMetadata`, `PostprocessConfig`, `Boundary`, `CountResult`,
  `FrameResult`, and typed status/error values.
- [ ] Keep inference tensor layout explicit: NCHW, RGB, FP32, ImageNet mean/std, 640×640.
- [ ] Load the model and construct the ONNX Runtime session once per camera session.
- [ ] Keep frame buffers and tensors reusable to minimize real-time allocations and copies.
- [ ] Implement temporal stabilization: rolling window, consensus count, hysteresis, and stable
  result lock.
- [ ] Add unit tests matching Python fixtures exactly within documented tolerances.
- [ ] Add golden parity tests for portrait, landscape, rotated, padded, horizontal, and vertical
  inputs.
- [ ] Expose a minimal JNI API to Kotlin without passing C++ exceptions across the boundary.
- [ ] Expose a minimal Objective-C++ wrapper to Swift without leaking C++ ownership details.
- [ ] Build Android ABIs required by product devices; do not ship unused ABIs in release builds.
- [ ] Build an iOS XCFramework for device and simulator architectures.
- [ ] Add AddressSanitizer and UndefinedBehaviorSanitizer test builds where supported.
- [ ] Map native status codes into Kotlin exceptions/results and Swift errors.

### ONNX Runtime integration

- [ ] Integrate the official ONNX Runtime C/C++ API into the shared core.
- [ ] Package the required ONNX Runtime native library and operators for each Android ABI.
- [ ] Integrate the official `onnxruntime-c` package into the iOS wrapper/core target.
- [ ] Start with CPU/XNNPACK and measure before testing NNAPI or Core ML execution providers.
- [ ] Verify session creation, tensor names/shapes, external model data loading, and shutdown on both
  platforms.
- [ ] Compare binary size, memory, latency, and accuracy before enabling an accelerator.

## Phase 5 — Native Android application

- [x] Create `apps/android` as a Kotlin/Jetpack Compose application.
- [x] Implement camera permission, preview, lifecycle, pause/resume, and error states.
- [x] Use CameraX `ImageAnalysis` with `STRATEGY_KEEP_ONLY_LATEST` so inference never queues stale
  frames: <https://developer.android.com/media/camera/camerax/analyze>.
- [x] Convert camera RGBA frames consistently and apply rotation before inference.
- [x] Add a single-stack framing guide and explicit `Move closer`, `Show full stack`, and
  `Hold steady` guidance.
- [x] Load the versioned ONNX/ORT model once per session.
- [ ] Integrate the shared C++ core and ONNX Runtime native library with CMake/Gradle.
- [ ] Integrate the JNI binding and verify tensor names, shapes, and buffer ownership.
- [x] Map boundaries from model/ROI coordinates into preview coordinates.
- [x] Draw numbered separator overlays and the stabilized count in real time.
- [x] Throttle inference to an initial device rate; continue showing camera preview at
  full frame rate.
- [x] Require several consistent frames before presenting a locked count.
- [x] Add rescan, confirm, and human-review actions.
- [ ] Record optional, privacy-approved failure examples only with operator consent.
- [ ] Add unit tests, screenshot/UI tests, instrumentation tests, and physical-device benchmarks.
- [ ] Test lifecycle interruption, screen rotation, thermal throttling, memory pressure, and
  camera-denied states.
- [ ] Produce signed internal QA builds before any store/distribution work.

## Phase 6 — Native iOS application

- [ ] Create `apps/ios` as a SwiftUI application.
- [ ] Implement camera permission, preview, lifecycle, interruption, and error states.
- [ ] Use `AVCaptureVideoDataOutput` on a serial callback queue and discard late frames when the
  inference pipeline is busy: <https://developer.apple.com/documentation/avfoundation/avcapturevideodataoutput>.
- [ ] Normalize camera orientation and mirroring before preprocessing.
- [ ] Add the same single-stack framing and operator guidance as Android.
- [ ] Integrate the shared C++ core and `onnxruntime-c` through an Objective-C++ wrapper.
- [ ] Integrate the core as an XCFramework or an internal framework target usable from Swift.
- [ ] Map boundaries from model/ROI coordinates into preview coordinates.
- [ ] Draw numbered separator overlays and the stabilized count in real time.
- [ ] Reuse the same confidence, stabilization, rescan, and human-review policy as Android.
- [ ] Add XCTest unit, UI, and golden-result tests.
- [ ] Benchmark representative older and current iPhones for latency, memory, battery, and heat.
- [ ] Test interruptions, background/foreground transitions, rotation, camera denial, and memory
  warnings.
- [ ] Produce signed TestFlight/internal QA builds before release preparation.

## Phase 7 — Cross-platform parity and real-time behavior

- [ ] Define one canonical preprocessing and coordinate-transform specification.
- [ ] Run identical approved golden images through Python, Android, and iOS.
- [ ] Require equivalent logits within numeric tolerance and identical final integer counts.
- [ ] Require boundary positions to match within a documented pixel tolerance.
- [ ] Test FP32/FP16/INT8 differences separately on each device class.
- [ ] Add temporal test sequences for camera motion, blur, stack entry/exit, and changing exposure.
- [ ] Ensure stale frames cannot update the UI after a newer result.
- [ ] Measure end-to-end latency from frame timestamp to overlay, not inference alone.
- [ ] Display `Scanning` until the count is stable; never present a fluctuating number as final.
- [ ] Keep a visible manual-QC route for low confidence and unsupported scenes.

## Phase 8 — Pilot, monitoring, and release

- [ ] Run shadow mode: the app predicts, but human QC remains authoritative.
- [ ] Compare predictions with actual packed quantities by shift, device, product, and count range.
- [ ] Review every high-confidence wrong result before increasing automation.
- [ ] Set thresholds from pilot evidence, not from the public dataset.
- [ ] Version the application, model, C++ core, thresholds, and dataset together.
- [ ] Add a signed model-update and rollback process if models can change independently of the app.
- [ ] Define offline logs that contain no images by default.
- [ ] Document incident response for systematic undercount or overcount.
- [ ] Train operators on framing, rescan, and human-review behavior.
- [ ] Obtain QC sign-off before the prediction can replace any manual count.

## CI and repository hygiene

- [ ] Add Python formatting, unit tests, ONNX export, and ONNX parity checks to CI.
- [ ] Add ClangFormat, Clang-Tidy, C++ unit tests, sanitizers, and Android/iOS cross-build checks.
- [ ] Add Android lint, unit tests, instrumentation smoke build, and release-size reporting.
- [ ] Add iOS build, unit tests, and simulator smoke tests on macOS CI.
- [x] During the prototype phase, commit only the approved files placed manually in
  `models/source/` and `models/exported/`; continue keeping generated run checkpoints out of Git.
- [ ] Move model distribution to Git LFS, a model registry, or release artifacts before model
  versions begin changing frequently.
- [ ] Add licenses and attribution for every public dataset and dependency.
- [ ] Never commit Roboflow, signing, store, or CI secrets.

## Immediate next actions

1. [ ] Collect 50–100 unseen factory images with verified counts.
2. [ ] Run the current checkpoint on them and save a factory-baseline CSV plus error overlays.
3. [ ] Decide the single-stack framing/ROI contract with QC operators.
4. [ ] Implement validation-only post-processing parameter search and count-based checkpoint
   selection.
5. [ ] Annotate factory boundaries for the most representative and most difficult images.
6. [ ] Fine-tune on factory data and evaluate once on the locked factory test split.
7. [ ] Lock and validate an ONNX artifact plus model manifest.
8. [ ] Build the C++ inference/post-processing core and its first Python parity fixtures.
9. [ ] Build a still-image Android ONNX Runtime proof of concept before real-time camera work.
10. [ ] Build the equivalent still-image iOS proof of concept and compare outputs.
