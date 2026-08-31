package com.bookspine.detector.inference

import com.bookspine.detector.domain.BoundaryOrientation
import java.nio.FloatBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ProfilePostprocessorTest {
    private val processor = ProfilePostprocessor()
    private val fullFrame = Letterbox(
        scale = 1f,
        padLeft = 0,
        padTop = 0,
        resizedWidth = ModelSpec.INPUT_SIZE,
        resizedHeight = ModelSpec.INPUT_SIZE,
    )

    @Test
    fun countsVerticalSeparatorBoundaries() {
        val logits = separatorLogits(
            orientation = BoundaryOrientation.VERTICAL,
            positions = listOf(30, 140, 250, 360, 470, 580),
        )

        val result = processor.process(logits, fullFrame, 640, 640, 12)

        assertEquals(BoundaryOrientation.VERTICAL, result.orientation)
        assertEquals(5, result.count)
        assertEquals(6, result.boundaries.size)
        assertTrue(result.confidence > 0.8f)
    }

    @Test
    fun countsHorizontalSeparatorBoundaries() {
        val logits = separatorLogits(
            orientation = BoundaryOrientation.HORIZONTAL,
            positions = listOf(50, 210, 370, 530),
        )

        val result = processor.process(logits, fullFrame, 640, 640, 12)

        assertEquals(BoundaryOrientation.HORIZONTAL, result.orientation)
        assertEquals(3, result.count)
    }

    private fun separatorLogits(
        orientation: BoundaryOrientation,
        positions: List<Int>,
    ): FloatBuffer {
        val size = ModelSpec.INPUT_SIZE
        val planeSize = size * size
        val values = FloatArray(planeSize * 2) { index ->
            if (index < planeSize) 0f else -10f
        }
        positions.forEach { position ->
            for (offset in -1..1) {
                if (orientation == BoundaryOrientation.VERTICAL) {
                    for (y in 0 until size) values[planeSize + y * size + position + offset] = 10f
                } else {
                    for (x in 0 until size) values[planeSize + (position + offset) * size + x] = 10f
                }
            }
        }
        return FloatBuffer.wrap(values)
    }
}
