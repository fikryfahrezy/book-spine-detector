package com.bookspine.detector.scan

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bookspine.detector.domain.BoundaryOrientation
import com.bookspine.detector.domain.FrameResult
import com.bookspine.detector.domain.ScanGuidance
import com.bookspine.detector.domain.guidance
import com.bookspine.detector.ui.theme.AppColors
import com.bookspine.detector.ui.theme.AppRadius
import com.bookspine.detector.ui.theme.AppSpace
import java.util.Locale
import kotlin.math.max

@Composable
fun ScanUi(
    state: ScanUiState,
    hasCameraPermission: Boolean,
    onRequestPermission: () -> Unit,
    onTogglePause: () -> Unit,
    onRescan: () -> Unit,
    onConfirm: () -> Unit,
    onHumanReview: () -> Unit,
) {
    Box(Modifier.fillMaxSize()) {
        if (hasCameraPermission) {
            ScanOverlay(result = state.lockedResult ?: state.latestResult)
        } else {
            Box(Modifier.fillMaxSize().background(AppColors.Ink))
        }

        AppHeader(
            paused = state.isPaused,
            enabled = hasCameraPermission && state.modelState == ModelState.READY,
            onTogglePause = onTogglePause,
        )

        when {
            !hasCameraPermission -> PermissionCard(onRequestPermission)
            state.modelState == ModelState.FAILED -> ErrorCard(
                title = "Model unavailable",
                message = state.errorMessage ?: "The on-device model could not be opened.",
            )
            else -> ControlPanel(
                state = state,
                onRescan = onRescan,
                onConfirm = onConfirm,
                onHumanReview = onHumanReview,
            )
        }
    }
}

@Composable
private fun AppHeader(paused: Boolean, enabled: Boolean, onTogglePause: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .padding(horizontal = AppSpace.Lg, vertical = AppSpace.Md),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column {
            Text(
                text = "BOOK COUNTER",
                color = AppColors.Accent,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.6.sp,
            )
            Text(
                text = "Scan one stack",
                color = AppColors.OnDark,
                style = MaterialTheme.typography.headlineMedium,
            )
        }
        OutlinedButton(
            onClick = onTogglePause,
            enabled = enabled,
            shape = CircleShape,
            border = androidx.compose.foundation.BorderStroke(1.dp, Color.White.copy(alpha = 0.35f)),
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = AppColors.OnDark,
                disabledContentColor = AppColors.OnDark.copy(alpha = 0.35f),
            ),
        ) {
            Text(if (paused) "Resume" else "Pause")
        }
    }
}

@Composable
private fun ScanOverlay(result: FrameResult?) {
    Canvas(Modifier.fillMaxSize().testTag("scan-overlay")) {
        val horizontalMargin = 24.dp.toPx()
        val guideTop = 112.dp.toPx()
        val guideBottom = size.height - 300.dp.toPx()
        val guide = Rect(horizontalMargin, guideTop, size.width - horizontalMargin, guideBottom)

        drawRect(AppColors.Scrim, size = androidx.compose.ui.geometry.Size(size.width, guide.top))
        drawRect(
            AppColors.Scrim,
            topLeft = Offset(0f, guide.bottom),
            size = androidx.compose.ui.geometry.Size(size.width, size.height - guide.bottom),
        )
        drawRect(
            AppColors.Scrim,
            topLeft = Offset(0f, guide.top),
            size = androidx.compose.ui.geometry.Size(guide.left, guide.height),
        )
        drawRect(
            AppColors.Scrim,
            topLeft = Offset(guide.right, guide.top),
            size = androidx.compose.ui.geometry.Size(size.width - guide.right, guide.height),
        )

        val corner = 38.dp.toPx()
        val path = Path().apply {
            moveTo(guide.left, guide.top + corner); lineTo(guide.left, guide.top); lineTo(guide.left + corner, guide.top)
            moveTo(guide.right - corner, guide.top); lineTo(guide.right, guide.top); lineTo(guide.right, guide.top + corner)
            moveTo(guide.right, guide.bottom - corner); lineTo(guide.right, guide.bottom); lineTo(guide.right - corner, guide.bottom)
            moveTo(guide.left + corner, guide.bottom); lineTo(guide.left, guide.bottom); lineTo(guide.left, guide.bottom - corner)
        }
        drawPath(path, AppColors.Accent, style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round))

        if (result == null) return@Canvas
        val previewScale = max(size.width / result.frameWidth, size.height / result.frameHeight)
        val offsetX = (size.width - result.frameWidth * previewScale) / 2f
        val offsetY = (size.height - result.frameHeight * previewScale) / 2f
        val lineColor = if (result.stable) AppColors.Success else AppColors.Warning
        clipRect(guide.left, guide.top, guide.right, guide.bottom) {
            if (result.orientation == BoundaryOrientation.VERTICAL) {
                val visiblePositions = result.boundaries
                    .map { offsetX + it.position * result.frameWidth * previewScale }
                    .filter { it in guide.left..guide.right }
                visiblePositions.forEachIndexed { index, x ->
                    drawLine(lineColor, Offset(x, guide.top), Offset(x, guide.bottom), 2.dp.toPx())
                    drawBoundaryNumber(index + 1, Offset(x, guide.top + 18.dp.toPx()), lineColor)
                }
            } else {
                val visiblePositions = result.boundaries
                    .map { offsetY + it.position * result.frameHeight * previewScale }
                    .filter { it in guide.top..guide.bottom }
                visiblePositions.forEachIndexed { index, y ->
                    drawLine(lineColor, Offset(guide.left, y), Offset(guide.right, y), 2.dp.toPx())
                    drawBoundaryNumber(index + 1, Offset(guide.left + 18.dp.toPx(), y), lineColor)
                }
            }
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawBoundaryNumber(
    number: Int,
    center: Offset,
    color: Color,
) {
    drawCircle(color, radius = 11.dp.toPx(), center = center)
    drawContext.canvas.nativeCanvas.drawText(
        number.toString(),
        center.x,
        center.y + 4.dp.toPx(),
        android.graphics.Paint().apply {
            this.color = android.graphics.Color.rgb(11, 18, 32)
            textSize = 11.sp.toPx()
            textAlign = android.graphics.Paint.Align.CENTER
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        },
    )
}

@Composable
private fun BoxScope.ControlPanel(
    state: ScanUiState,
    onRescan: () -> Unit,
    onConfirm: () -> Unit,
    onHumanReview: () -> Unit,
) {
    val result = state.lockedResult ?: state.latestResult
    val guidance = when {
        state.isPaused && state.lockedResult == null -> ScanGuidance.HOLD_STEADY
        else -> result?.guidance() ?: ScanGuidance.READY
    }
    Surface(
        modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth(),
        color = AppColors.Surface,
        contentColor = AppColors.TextPrimary,
        shape = RoundedCornerShape(topStart = AppRadius.Large, topEnd = AppRadius.Large),
        tonalElevation = 0.dp,
    ) {
        Column(
            Modifier
                .navigationBarsPadding()
                .padding(horizontal = AppSpace.Lg, vertical = AppSpace.Lg),
        ) {
            when (state.decision) {
                OperatorDecision.CONFIRMED -> DecisionContent(
                    title = "Count confirmed",
                    detail = "This stack has been accepted for the current QC step.",
                    color = AppColors.Success,
                    onRescan = onRescan,
                )
                OperatorDecision.NEEDS_REVIEW -> DecisionContent(
                    title = "Marked for review",
                    detail = "Keep this stack aside for a manual count.",
                    color = AppColors.Warning,
                    onRescan = onRescan,
                )
                OperatorDecision.NONE -> ActiveScanContent(
                    state = state,
                    result = result,
                    guidance = guidance,
                    onRescan = onRescan,
                    onConfirm = onConfirm,
                    onHumanReview = onHumanReview,
                )
            }
        }
    }
}

@Composable
private fun ActiveScanContent(
    state: ScanUiState,
    result: FrameResult?,
    guidance: ScanGuidance,
    onRescan: () -> Unit,
    onConfirm: () -> Unit,
    onHumanReview: () -> Unit,
) {
    val locked = state.lockedResult != null
    Row(verticalAlignment = Alignment.Top) {
        Column(Modifier.weight(1f)) {
            StatusPill(
                label = when {
                    state.modelState == ModelState.LOADING -> "PREPARING"
                    locked -> "LOCKED"
                    state.isPaused -> "PAUSED"
                    else -> "SCANNING"
                },
                color = if (locked) AppColors.Success else AppColors.Warning,
            )
            Spacer(Modifier.height(AppSpace.Sm))
            Text(guidance.title, style = MaterialTheme.typography.headlineMedium)
            Text(
                guidance.detail,
                color = AppColors.TextSecondary,
                style = MaterialTheme.typography.bodyLarge,
            )
        }
        if (state.modelState == ModelState.LOADING) {
            CircularProgressIndicator(
                Modifier.padding(top = AppSpace.Md).size(36.dp),
                color = AppColors.AccentStrong,
                strokeWidth = 3.dp,
            )
        } else if (locked) {
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = result?.count?.toString() ?: "—",
                    modifier = Modifier.testTag("stable-count"),
                    color = AppColors.TextPrimary,
                    fontSize = 64.sp,
                    lineHeight = 64.sp,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = "books · ${formatConfidence(result?.confidence)}",
                    color = AppColors.TextSecondary,
                    fontSize = 13.sp,
                )
            }
        }
    }

    state.errorMessage?.let {
        Spacer(Modifier.height(AppSpace.Md))
        Text(it, color = AppColors.Danger, fontSize = 13.sp)
    }
    Spacer(Modifier.height(AppSpace.Lg))
    if (locked) {
        Row(horizontalArrangement = Arrangement.spacedBy(AppSpace.Sm)) {
            OutlinedAction("Rescan", Modifier.weight(1f), onRescan)
            PrimaryAction("Confirm", Modifier.weight(1.35f), onConfirm)
        }
        Spacer(Modifier.height(AppSpace.Sm))
        OutlinedAction("Send to human review", Modifier.fillMaxWidth(), onHumanReview)
    } else {
        OutlinedAction("I can’t get a clear count", Modifier.fillMaxWidth(), onHumanReview)
    }
}

@Composable
private fun DecisionContent(title: String, detail: String, color: Color, onRescan: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(12.dp).clip(CircleShape).background(color))
        Spacer(Modifier.size(AppSpace.Sm))
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.headlineMedium)
            Text(detail, color = AppColors.TextSecondary)
        }
    }
    Spacer(Modifier.height(AppSpace.Lg))
    PrimaryAction("Scan next stack", Modifier.fillMaxWidth(), onRescan)
}

@Composable
private fun StatusPill(label: String, color: Color) {
    Row(
        modifier = Modifier
            .clip(CircleShape)
            .background(color.copy(alpha = 0.20f))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(7.dp).clip(CircleShape).background(color))
        Spacer(Modifier.size(6.dp))
        Text(label, color = AppColors.TextPrimary, fontSize = 11.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun BoxScope.PermissionCard(onRequestPermission: () -> Unit) {
    Surface(
        modifier = Modifier
            .align(Alignment.Center)
            .padding(AppSpace.Lg)
            .fillMaxWidth(),
        color = AppColors.Surface,
        shape = RoundedCornerShape(AppRadius.Large),
    ) {
        Column(
            Modifier.padding(AppSpace.Xl),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Camera access needed", style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.height(AppSpace.Sm))
            Text(
                "Frames stay on this device and are used only to count the visible stack.",
                color = AppColors.TextSecondary,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(AppSpace.Lg))
            PrimaryAction("Allow camera", Modifier.fillMaxWidth(), onRequestPermission)
        }
    }
}

@Composable
private fun BoxScope.ErrorCard(title: String, message: String) {
    Surface(
        modifier = Modifier.align(Alignment.Center).padding(AppSpace.Lg).fillMaxWidth(),
        color = AppColors.Surface,
        shape = RoundedCornerShape(AppRadius.Large),
    ) {
        Column(Modifier.padding(AppSpace.Xl)) {
            Text(title, style = MaterialTheme.typography.headlineMedium, color = AppColors.Danger)
            Spacer(Modifier.height(AppSpace.Sm))
            Text(message, color = AppColors.TextSecondary)
        }
    }
}

@Composable
private fun PrimaryAction(label: String, modifier: Modifier, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        modifier = modifier.height(52.dp),
        shape = RoundedCornerShape(AppRadius.Medium),
        colors = ButtonDefaults.buttonColors(
            containerColor = AppColors.Accent,
            contentColor = AppColors.Ink,
        ),
    ) {
        Text(label, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun OutlinedAction(label: String, modifier: Modifier, onClick: () -> Unit) {
    OutlinedButton(
        onClick = onClick,
        modifier = modifier.height(52.dp),
        shape = RoundedCornerShape(AppRadius.Medium),
        border = androidx.compose.foundation.BorderStroke(1.dp, AppColors.TextPrimary.copy(alpha = 0.18f)),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = AppColors.TextPrimary),
    ) {
        Text(label, fontWeight = FontWeight.SemiBold)
    }
}

private fun formatConfidence(confidence: Float?): String = if (confidence == null) {
    ""
} else {
    String.format(Locale.US, "%.0f%% confidence", confidence * 100)
}
