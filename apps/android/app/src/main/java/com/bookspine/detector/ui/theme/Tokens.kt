package com.bookspine.detector.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

/** Android mapping of apps/design-system/tokens.json. Keep names aligned for SwiftUI parity. */
object AppColors {
    val Ink = Color(0xFF0B1220)
    val Surface = Color(0xFFF7F8F3)
    val SurfaceRaised = Color(0xFFFFFFFF)
    val SurfaceMuted = Color(0xFFE9ECE5)
    val TextPrimary = Color(0xFF121A27)
    val TextSecondary = Color(0xFF5B6574)
    val Accent = Color(0xFFD8FF70)
    val AccentStrong = Color(0xFFB8E34D)
    val Success = Color(0xFF3ED598)
    val Warning = Color(0xFFFFBF69)
    val Danger = Color(0xFFFF6B6B)
    val OnDark = Color(0xFFF8FAF5)
    val Scrim = Color(0x99070B12)
}

object AppRadius {
    val Small = 10.dp
    val Medium = 18.dp
    val Large = 28.dp
}

object AppSpace {
    val Xs = 4.dp
    val Sm = 8.dp
    val Md = 16.dp
    val Lg = 24.dp
    val Xl = 32.dp
}
