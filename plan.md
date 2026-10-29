Act as a senior computer vision engineer. Improve the existing `person-reid-multicamera` project to achieve reliable, testable person detection, tracking, and long-term re-identification. Do not rebuild the project from scratch.

First inspect the entire repository, architecture, dependencies, configuration, model weights, and tests. Create a prioritized implementation plan, then execute the following phases sequentially, testing each phase before continuing.

**Phase 1 — ReID Backbone**
- Replace ImageNet-only ResNet50 features with configurable pretrained OSNet person-ReID embeddings; retain ResNet50 as a baseline.
- Support full-body and optional three-part embeddings using `USE_PART_EMBEDDING`.
- Preserve existing interfaces where possible; validate dimensions, finite values, L2 normalization, and model-loading errors.
- Remove legacy fallback imports and random embedding fallbacks.
- Recalibrate similarity thresholds only using validation evidence.

**Phase 2 — Tracking and Gallery**
- Improve motion/IoU gating and configurable association costs.
- Add ByteTrack-style low-confidence detection association before lost-track recovery.
- Support configurable gallery exemplars or EMA embeddings.
- Require configurable consecutive consistent matches before confirming revived tracks.
- Make gallery spatial gating and static-track pruning configurable.
- Preserve existing behavior by default wherever possible.

**Phase 3 — Evaluation**
- Create `evaluation/evaluate.py` to run the tracker on MOT-format ground-truth sequences or compatible video datasets.
- Generate MOT-format predictions and calculate IDF1, HOTA, MOTA, ID switches, fragmentation, and FPS using compatible evaluation tools such as TrackEval.
- Add ablation options for disabling the gallery and part embeddings, and selecting ResNet50 or OSNet.
- Save reproducible results in `evaluation/results.md`. Never fabricate metrics.

**Phase 4 — Documentation**
- Update README and CHANGELOG.
- Document configuration options, thresholds, model setup, dependencies, installation, and run commands.
- Document limitations, including camera-motion compensation and cross-camera support.
- Claim cross-camera ReID only if it is actually implemented and validated.

**Engineering requirements**
- Use explicit package imports and maintain modular, typed code.
- Never fabricate detections, embeddings, search results, or test outcomes.
- Handle invalid inputs, missing weights, inference failures, and incompatible embedding dimensions safely.
- Add unit, integration, and regression tests.
- Benchmark against a fixed baseline using identical videos and settings.
- Do not blindly tune thresholds or break existing functionality.
- Use a separate Git branch for each phase and keep the application runnable.
- Inspect and present the plan before editing core tracking code; wait for approval before Phase 1 changes.
- After approval, complete each phase sequentially and report changed files, commands executed, actual test results, measured improvements, and remaining issues.
- Do not claim production readiness without evidence from real tests and evaluation.

Begin by inspecting the repository and presenting the implementation plan. Do not modify files until I approve the plan.