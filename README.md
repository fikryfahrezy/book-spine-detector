# Book Spine Detector

Monorepo for a real-time mobile system that counts books in factory stacks.

## Repository layout

```text
book-spine-detector/
├── ml/                 # Python training, evaluation, export, and notebooks
├── apps/
│   ├── android/        # Native Android application (planned)
│   └── ios/            # Native iOS application (planned)
├── crates/             # Shared Rust inference/processing code (planned)
├── models/
│   ├── source/         # Curated PyTorch training checkpoints
│   └── exported/       # Curated mobile-ready model artifacts
└── docs/               # Architecture and product documentation
```

The current implementation is the ML training scaffold. See [ml/README.md](ml/README.md)
for dataset preparation, Colab training, evaluation, manual testing, and model export.

## Google Colab quick start

```python
!git clone https://github.com/YOUR_USERNAME/book-spine-detector.git
%cd /content/book-spine-detector/ml
!pip install -e ".[notebook]"
```

