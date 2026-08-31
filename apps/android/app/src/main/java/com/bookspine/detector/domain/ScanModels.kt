package com.bookspine.detector.domain

enum class BoundaryOrientation {
    VERTICAL,
    HORIZONTAL,
}

data class Boundary(
    /** Position in the oriented camera frame, normalized to 0..1. */
    val position: Float,
)

data class FrameResult(
    val count: Int,
    val orientation: BoundaryOrientation,
    val boundaries: List<Boundary>,
    val confidence: Float,
    val frameWidth: Int,
    val frameHeight: Int,
    val inferenceMillis: Long,
    val stable: Boolean = false,
)

enum class ScanGuidance(val title: String, val detail: String) {
    READY("Frame one complete stack", "Keep the first and last book inside the guide"),
    MOVE_CLOSER("Move closer", "Let the stack fill more of the frame"),
    SHOW_FULL_STACK("Show the full stack", "The first or last book may be outside the guide"),
    HOLD_STEADY("Hold steady", "Keep the phone still while the count settles"),
    LOCKED("Count ready", "Review the separators, then confirm"),
}

fun FrameResult.guidance(): ScanGuidance {
    if (stable) return ScanGuidance.LOCKED
    if (boundaries.isEmpty()) return ScanGuidance.MOVE_CLOSER

    val first = boundaries.first().position
    val last = boundaries.last().position
    if (first < 0.025f || last > 0.975f) return ScanGuidance.SHOW_FULL_STACK
    if (last - first < 0.45f) return ScanGuidance.MOVE_CLOSER
    return ScanGuidance.HOLD_STEADY
}
