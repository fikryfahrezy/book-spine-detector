package com.bookspine.detector.inference

import java.nio.ByteBuffer

data class RgbaFrame(
    val bytes: ByteBuffer,
    val width: Int,
    val height: Int,
    val rowStride: Int,
    val pixelStride: Int,
    val rotationDegrees: Int,
) {
    init {
        require(width > 0 && height > 0)
        require(pixelStride >= 4) { "Expected RGBA pixels" }
        require(rotationDegrees in setOf(0, 90, 180, 270))
    }

    val orientedWidth: Int
        get() = if (rotationDegrees == 90 || rotationDegrees == 270) height else width

    val orientedHeight: Int
        get() = if (rotationDegrees == 90 || rotationDegrees == 270) width else height
}
