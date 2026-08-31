package com.bookspine.detector.inference

import java.nio.FloatBuffer
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt

internal data class Letterbox(
    val scale: Float,
    val padLeft: Int,
    val padTop: Int,
    val resizedWidth: Int,
    val resizedHeight: Int,
)

internal class ImagePreprocessor {
    fun write(frame: RgbaFrame, destination: FloatBuffer): Letterbox {
        val size = ModelSpec.INPUT_SIZE
        val planeSize = size * size
        require(destination.capacity() >= planeSize * 3)

        // The Python pipeline letterboxes with black before ImageNet normalization.
        for (channel in 0..2) {
            val black = (0f - ModelSpec.mean[channel]) / ModelSpec.std[channel]
            val offset = channel * planeSize
            for (index in 0 until planeSize) destination.put(offset + index, black)
        }

        val orientedWidth = frame.orientedWidth
        val orientedHeight = frame.orientedHeight
        val scale = min(size.toFloat() / orientedWidth, size.toFloat() / orientedHeight)
        val resizedWidth = (orientedWidth * scale).roundToInt().coerceAtLeast(1)
        val resizedHeight = (orientedHeight * scale).roundToInt().coerceAtLeast(1)
        val padLeft = (size - resizedWidth) / 2
        val padTop = (size - resizedHeight) / 2

        for (targetY in 0 until resizedHeight) {
            val sourceY = ((targetY + 0.5f) / scale - 0.5f).coerceIn(0f, orientedHeight - 1f)
            for (targetX in 0 until resizedWidth) {
                val sourceX = ((targetX + 0.5f) / scale - 0.5f).coerceIn(0f, orientedWidth - 1f)
                val pixelIndex = (targetY + padTop) * size + targetX + padLeft
                for (channel in 0..2) {
                    val value = bilinearChannel(frame, sourceX, sourceY, channel)
                    val normalized = (value / 255f - ModelSpec.mean[channel]) /
                        ModelSpec.std[channel]
                    destination.put(channel * planeSize + pixelIndex, normalized)
                }
            }
        }
        destination.position(0)
        destination.limit(planeSize * 3)
        return Letterbox(scale, padLeft, padTop, resizedWidth, resizedHeight)
    }

    private fun bilinearChannel(frame: RgbaFrame, x: Float, y: Float, channel: Int): Float {
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val x1 = (x0 + 1).coerceAtMost(frame.orientedWidth - 1)
        val y1 = (y0 + 1).coerceAtMost(frame.orientedHeight - 1)
        val xWeight = x - x0
        val yWeight = y - y0
        val top = sample(frame, x0, y0, channel) * (1f - xWeight) +
            sample(frame, x1, y0, channel) * xWeight
        val bottom = sample(frame, x0, y1, channel) * (1f - xWeight) +
            sample(frame, x1, y1, channel) * xWeight
        return top * (1f - yWeight) + bottom * yWeight
    }

    private fun sample(frame: RgbaFrame, orientedX: Int, orientedY: Int, channel: Int): Float {
        val (sourceX, sourceY) = when (frame.rotationDegrees) {
            0 -> orientedX to orientedY
            90 -> orientedY to (frame.height - 1 - orientedX)
            180 -> (frame.width - 1 - orientedX) to (frame.height - 1 - orientedY)
            270 -> (frame.width - 1 - orientedY) to orientedX
            else -> error("Unsupported rotation")
        }
        val index = frame.bytes.position() + sourceY * frame.rowStride +
            sourceX * frame.pixelStride + channel
        return (frame.bytes.get(index).toInt() and 0xff).toFloat()
    }
}
