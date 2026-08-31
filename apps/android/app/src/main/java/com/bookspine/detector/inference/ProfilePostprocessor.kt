package com.bookspine.detector.inference

import com.bookspine.detector.domain.Boundary
import com.bookspine.detector.domain.BoundaryOrientation
import com.bookspine.detector.domain.FrameResult
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max

internal class ProfilePostprocessor {
    fun process(
        logits: FloatBuffer,
        letterbox: Letterbox,
        frameWidth: Int,
        frameHeight: Int,
        inferenceMillis: Long,
    ): FrameResult {
        val size = ModelSpec.INPUT_SIZE
        val planeSize = size * size
        require(logits.remaining() >= planeSize * ModelSpec.CLASS_COUNT) {
            "Unexpected model output size: ${logits.remaining()}"
        }

        val vertical = FloatArray(letterbox.resizedWidth)
        val horizontal = FloatArray(letterbox.resizedHeight)
        for (y in 0 until letterbox.resizedHeight) {
            for (x in 0 until letterbox.resizedWidth) {
                val modelIndex = (y + letterbox.padTop) * size + x + letterbox.padLeft
                val backgroundLogit = logits.get(modelIndex)
                val separatorLogit = logits.get(planeSize + modelIndex)
                val probability = (1.0 / (1.0 + exp((backgroundLogit - separatorLogit).toDouble())))
                    .toFloat()
                if (probability >= ModelSpec.PIXEL_THRESHOLD) {
                    vertical[x] += probability
                    horizontal[y] += probability
                }
            }
        }
        for (x in vertical.indices) vertical[x] /= letterbox.resizedHeight
        for (y in horizontal.indices) horizontal[y] /= letterbox.resizedWidth

        val verticalCandidate = candidate(vertical, BoundaryOrientation.VERTICAL)
        val horizontalCandidate = candidate(horizontal, BoundaryOrientation.HORIZONTAL)
        // `max` in Python keeps the first candidate when scores tie.
        val selected = if (horizontalCandidate.score > verticalCandidate.score) {
            horizontalCandidate
        } else {
            verticalCandidate
        }
        val denominator = if (selected.orientation == BoundaryOrientation.VERTICAL) {
            letterbox.resizedWidth
        } else {
            letterbox.resizedHeight
        }.coerceAtLeast(1)

        return FrameResult(
            count = max(0, selected.peaks.size - 1),
            orientation = selected.orientation,
            boundaries = selected.peaks.map { Boundary(it.toFloat() / denominator) },
            confidence = selected.confidence,
            frameWidth = frameWidth,
            frameHeight = frameHeight,
            inferenceMillis = inferenceMillis,
        )
    }

    private fun candidate(values: FloatArray, orientation: BoundaryOrientation): Candidate {
        val profile = normalize(smooth(values, ModelSpec.SMOOTHING_WINDOW))
        val peaks = findPeaks(profile)
        val prominenceMean = peaks.map { it.prominence }.average().takeUnless { it.isNaN() } ?: 0.0
        val heightMean = peaks.map { it.height }.average().takeUnless { it.isNaN() } ?: 0.0
        return Candidate(
            orientation = orientation,
            peaks = peaks.map(Peak::index),
            score = prominenceMean.toFloat() * peaks.size.coerceAtMost(8),
            confidence = (0.55 * heightMean + 0.45 * prominenceMean).toFloat().coerceIn(0f, 1f),
        )
    }

    private fun smooth(values: FloatArray, window: Int): FloatArray {
        if (window <= 1) return values.copyOf()
        val leftRadius = window / 2
        val rightRadius = window - leftRadius - 1
        return FloatArray(values.size) { index ->
            var sum = 0f
            for (offset in -leftRadius..rightRadius) {
                sum += values[(index + offset).coerceIn(values.indices)]
            }
            sum / window
        }
    }

    private fun normalize(values: FloatArray): FloatArray {
        if (values.isEmpty()) return values
        val sorted = values.sortedArray()
        val low = percentile(sorted, 0.10f)
        val high = percentile(sorted, 0.995f)
        if (high <= low + 1e-8f) return FloatArray(values.size)
        return FloatArray(values.size) { ((values[it] - low) / (high - low)).coerceIn(0f, 1f) }
    }

    private fun percentile(sorted: FloatArray, quantile: Float): Float {
        val position = (sorted.size - 1) * quantile
        val lower = position.toInt()
        val upper = (lower + 1).coerceAtMost(sorted.lastIndex)
        val fraction = position - lower
        return sorted[lower] * (1f - fraction) + sorted[upper] * fraction
    }

    private fun findPeaks(profile: FloatArray): List<Peak> {
        val candidates = mutableListOf<Peak>()
        var index = 1
        while (index < profile.lastIndex) {
            if (profile[index] <= profile[index - 1]) {
                index++
                continue
            }
            var plateauEnd = index
            while (plateauEnd < profile.lastIndex && profile[plateauEnd] == profile[plateauEnd + 1]) {
                plateauEnd++
            }
            if (profile[plateauEnd] > profile.getOrElse(plateauEnd + 1) { Float.POSITIVE_INFINITY }) {
                val peakIndex = (index + plateauEnd) / 2
                val height = profile[peakIndex]
                val prominence = prominence(profile, peakIndex)
                if (height >= ModelSpec.PEAK_HEIGHT && prominence >= ModelSpec.PEAK_PROMINENCE) {
                    candidates += Peak(peakIndex, height, prominence)
                }
            }
            index = plateauEnd + 1
        }

        val accepted = mutableListOf<Peak>()
        for (candidate in candidates.sortedByDescending(Peak::height)) {
            if (accepted.none { kotlin.math.abs(it.index - candidate.index) < ModelSpec.MINIMUM_PEAK_DISTANCE }) {
                accepted += candidate
            }
        }
        return accepted.sortedBy(Peak::index)
    }

    private fun prominence(profile: FloatArray, peak: Int): Float {
        val peakHeight = profile[peak]
        var leftMinimum = peakHeight
        var cursor = peak - 1
        while (cursor >= 0 && profile[cursor] <= peakHeight) {
            leftMinimum = minOf(leftMinimum, profile[cursor])
            cursor--
        }
        var rightMinimum = peakHeight
        cursor = peak + 1
        while (cursor < profile.size && profile[cursor] <= peakHeight) {
            rightMinimum = minOf(rightMinimum, profile[cursor])
            cursor++
        }
        return peakHeight - maxOf(leftMinimum, rightMinimum)
    }

    private data class Peak(val index: Int, val height: Float, val prominence: Float)

    private data class Candidate(
        val orientation: BoundaryOrientation,
        val peaks: List<Int>,
        val score: Float,
        val confidence: Float,
    )
}
