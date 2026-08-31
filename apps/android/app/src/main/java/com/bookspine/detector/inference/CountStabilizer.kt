package com.bookspine.detector.inference

import com.bookspine.detector.domain.FrameResult
import java.util.ArrayDeque

class CountStabilizer(
    private val windowSize: Int = 5,
    private val requiredAgreement: Int = 4,
    private val minimumConfidence: Float = 0.55f,
) {
    private val window = ArrayDeque<FrameResult>()

    init {
        require(requiredAgreement in 1..windowSize)
    }

    @Synchronized
    fun add(result: FrameResult): FrameResult? {
        if (result.confidence < minimumConfidence) {
            window.clear()
            return null
        }
        window.addLast(result)
        while (window.size > windowSize) window.removeFirst()
        if (window.size < requiredAgreement) return null

        val agreement = window.filter {
            it.count == result.count && it.orientation == result.orientation
        }
        if (agreement.size < requiredAgreement) return null
        val confidence = agreement.map(FrameResult::confidence).average().toFloat()
        return result.copy(confidence = confidence, stable = true)
    }

    @Synchronized
    fun reset() = window.clear()
}
