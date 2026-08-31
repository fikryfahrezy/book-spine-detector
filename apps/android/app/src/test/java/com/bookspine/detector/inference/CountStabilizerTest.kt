package com.bookspine.detector.inference

import com.bookspine.detector.domain.BoundaryOrientation
import com.bookspine.detector.domain.FrameResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CountStabilizerTest {
    @Test
    fun locksOnlyAfterFourMatchingConfidentFrames() {
        val stabilizer = CountStabilizer()

        repeat(3) { assertNull(stabilizer.add(result(count = 8))) }
        val locked = stabilizer.add(result(count = 8))

        assertEquals(8, locked?.count)
        assertTrue(locked?.stable == true)
    }

    @Test
    fun lowConfidenceClearsAgreementWindow() {
        val stabilizer = CountStabilizer()
        repeat(3) { stabilizer.add(result(count = 8)) }

        assertNull(stabilizer.add(result(count = 8, confidence = 0.4f)))
        val next = stabilizer.add(result(count = 8))

        assertNull(next)
        assertFalse(next?.stable == true)
    }

    private fun result(count: Int, confidence: Float = 0.9f) = FrameResult(
        count = count,
        orientation = BoundaryOrientation.VERTICAL,
        boundaries = emptyList(),
        confidence = confidence,
        frameWidth = 640,
        frameHeight = 640,
        inferenceMillis = 20,
    )
}
