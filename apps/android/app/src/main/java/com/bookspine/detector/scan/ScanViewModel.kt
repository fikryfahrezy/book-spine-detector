package com.bookspine.detector.scan

import android.app.Application
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.bookspine.detector.domain.FrameResult
import com.bookspine.detector.inference.CountStabilizer
import com.bookspine.detector.inference.RgbaFrame
import com.bookspine.detector.inference.SeparatorModelBinding
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

enum class ModelState { LOADING, READY, FAILED }
enum class OperatorDecision { NONE, CONFIRMED, NEEDS_REVIEW }

data class ScanUiState(
    val modelState: ModelState = ModelState.LOADING,
    val latestResult: FrameResult? = null,
    val lockedResult: FrameResult? = null,
    val isPaused: Boolean = false,
    val decision: OperatorDecision = OperatorDecision.NONE,
    val errorMessage: String? = null,
)

class ScanViewModel(application: Application) : AndroidViewModel(application) {
    private val mutableState = MutableStateFlow(ScanUiState())
    val state: StateFlow<ScanUiState> = mutableState.asStateFlow()

    private val stabilizer = CountStabilizer()
    private val lastInferenceAt = AtomicLong(0L)
    @Volatile private var binding: SeparatorModelBinding? = null

    init {
        viewModelScope.launch(Dispatchers.Default) {
            runCatching { SeparatorModelBinding.create(getApplication()) }
                .onSuccess { created ->
                    binding = created
                    mutableState.update { it.copy(modelState = ModelState.READY) }
                }
                .onFailure { error ->
                    mutableState.update {
                        it.copy(
                            modelState = ModelState.FAILED,
                            errorMessage = error.message ?: "The model could not be loaded.",
                        )
                    }
                }
        }
    }

    /** Called synchronously on CameraX's single analysis executor. */
    fun analyze(image: ImageProxy) {
        val snapshot = mutableState.value
        val model = binding ?: return
        if (snapshot.isPaused || snapshot.lockedResult != null || snapshot.decision != OperatorDecision.NONE) {
            return
        }
        val now = System.nanoTime()
        val previous = lastInferenceAt.get()
        if (previous != 0L && now - previous < INFERENCE_INTERVAL_NANOS) return
        if (!lastInferenceAt.compareAndSet(previous, now)) return

        val plane = image.planes.singleOrNull() ?: return
        val frame = RgbaFrame(
            bytes = plane.buffer,
            width = image.width,
            height = image.height,
            rowStride = plane.rowStride,
            pixelStride = plane.pixelStride,
            rotationDegrees = image.imageInfo.rotationDegrees,
        )
        runCatching { model.run(frame) }
            .onSuccess { raw ->
                val stable = stabilizer.add(raw)
                mutableState.update {
                    it.copy(
                        latestResult = stable ?: raw,
                        lockedResult = stable,
                        errorMessage = null,
                    )
                }
            }
            .onFailure { error ->
                mutableState.update {
                    it.copy(errorMessage = error.message ?: "Inference failed. Try rescanning.")
                }
            }
    }

    fun togglePause() = mutableState.update { state ->
        state.copy(isPaused = !state.isPaused)
    }

    fun rescan() {
        stabilizer.reset()
        lastInferenceAt.set(0L)
        mutableState.update {
            it.copy(
                latestResult = null,
                lockedResult = null,
                isPaused = false,
                decision = OperatorDecision.NONE,
                errorMessage = null,
            )
        }
    }

    fun confirm() = mutableState.update {
        if (it.lockedResult == null) it else it.copy(
            isPaused = true,
            decision = OperatorDecision.CONFIRMED,
        )
    }

    fun requestHumanReview() = mutableState.update {
        it.copy(isPaused = true, decision = OperatorDecision.NEEDS_REVIEW)
    }

    fun reportCameraError(error: Throwable) = mutableState.update {
        it.copy(errorMessage = error.message ?: "The camera could not be started.")
    }

    override fun onCleared() {
        binding?.close()
        binding = null
        super.onCleared()
    }

    private companion object {
        // The preview stays fluid; expensive inference is capped at four attempts per second.
        const val INFERENCE_INTERVAL_NANOS = 250_000_000L
    }
}
