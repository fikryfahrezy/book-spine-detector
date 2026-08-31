# Book Spine Detector

Monorepo for a real-time mobile system that counts books in factory stacks.

## Repository layout

```text
book-spine-detector/
├── ml/                 # Python training, evaluation, export, and notebooks
├── apps/
│   ├── android/        # Native Android application (planned)
│   └── ios/            # Native iOS application (planned)
├── core/               # Shared C++ inference/processing library (planned)
├── models/
│   ├── source/         # Curated PyTorch training checkpoints
│   └── exported/       # Curated mobile-ready model artifacts
└── docs/               # Architecture and product documentation
```

The repository includes the ML training scaffold and an offline native Android scanner. See
[ml/README.md](ml/README.md) for dataset preparation, Colab training, evaluation, manual testing,
and model export. See [apps/android/README.md](apps/android/README.md) for the Android build,
architecture, and device-verification checklist.
See [docs/TODO.md](docs/TODO.md) for the model-improvement and native mobile roadmap.

## Google Colab quick start

```python
!git clone https://github.com/YOUR_USERNAME/book-spine-detector.git
%cd /content/book-spine-detector/ml
!pip install -e ".[notebook]"
```
