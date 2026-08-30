# Dataset layout

Generated data is intentionally ignored by Git. The training loader expects:

```text
data/processed/public_books/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Every mask must have the same stem as its image. Mask value `0` is background and any
non-zero value is a separator or long book edge.

The `book-spine-prepare-yolo` command converts common Roboflow/YOLO polygon, oriented-box,
and ordinary-box exports into this representation.

