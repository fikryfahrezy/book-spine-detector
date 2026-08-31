package com.bookspine.detector.inference

internal object ModelSpec {
    const val MODEL_FILE = "separator_model.onnx"
    const val EXTERNAL_DATA_FILE = "separator_model.onnx.data"
    const val INPUT_NAME = "image"
    const val OUTPUT_NAME = "separator_logits"
    const val INPUT_SIZE = 640
    const val CLASS_COUNT = 2

    val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
    val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    const val PIXEL_THRESHOLD = 0.45f
    const val PEAK_HEIGHT = 0.30f
    const val PEAK_PROMINENCE = 0.08f
    const val SMOOTHING_WINDOW = 9
    const val MINIMUM_PEAK_DISTANCE = 8
}
