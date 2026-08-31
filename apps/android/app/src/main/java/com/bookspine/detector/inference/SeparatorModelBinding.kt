package com.bookspine.detector.inference

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.bookspine.detector.domain.FrameResult
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.system.measureNanoTime

/**
 * Narrow Android binding around ONNX Runtime. No ORT type crosses this class boundary, making the
 * same `FrameResult` contract straightforward to implement behind an Objective-C++/Swift adapter.
 */
class SeparatorModelBinding private constructor(
    private val environment: OrtEnvironment,
    private val session: OrtSession,
) : Closeable {
    private val preprocessor = ImagePreprocessor()
    private val postprocessor = ProfilePostprocessor()
    private val inputName = session.inputNames.single()
    private val outputName = session.outputNames.single()
    private val inputBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(3 * ModelSpec.INPUT_SIZE * ModelSpec.INPUT_SIZE * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

    init {
        require(inputName == ModelSpec.INPUT_NAME) { "Expected input '${ModelSpec.INPUT_NAME}', got '$inputName'" }
        require(outputName == ModelSpec.OUTPUT_NAME) {
            "Expected output '${ModelSpec.OUTPUT_NAME}', got '$outputName'"
        }
        val inputInfo = session.inputInfo.getValue(inputName).info as TensorInfo
        require(inputInfo.shape.matchesImageTensor(channels = 3)) {
            "Expected ${ModelSpec.inputShape.contentToString()}, got ${inputInfo.shape.contentToString()}"
        }
    }

    @Synchronized
    fun run(frame: RgbaFrame): FrameResult {
        val letterbox = preprocessor.write(frame, inputBuffer)
        var result: FrameResult? = null
        val elapsedNanos = measureNanoTime {
            OnnxTensor.createTensor(environment, inputBuffer, ModelSpec.inputShape).use { input ->
                session.run(mapOf(inputName to input)).use { outputs ->
                    val output = outputs.get(outputName).orElseThrow {
                        IllegalStateException("Model did not return $outputName")
                    } as OnnxTensor
                    val info = output.info
                    require(info.shape.matchesImageTensor(channels = 2)) {
                        "Unexpected output shape ${info.shape.contentToString()}"
                    }
                    result = postprocessor.process(
                        logits = output.floatBuffer,
                        letterbox = letterbox,
                        frameWidth = frame.orientedWidth,
                        frameHeight = frame.orientedHeight,
                        inferenceMillis = 0,
                    )
                }
            }
        }
        return checkNotNull(result).copy(inferenceMillis = elapsedNanos / 1_000_000)
    }

    override fun close() {
        session.close()
    }

    companion object {
        fun create(context: Context): SeparatorModelBinding {
            val model = ModelFiles.install(context.applicationContext)
            val environment = OrtEnvironment.getEnvironment("book-spine-detector")
            val options = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            return try {
                SeparatorModelBinding(environment, environment.createSession(model.absolutePath, options))
            } finally {
                options.close()
            }
        }
    }
}

/** Export uses a dynamic batch axis (`-1`); the mobile binding always executes a batch of one. */
private fun LongArray.matchesImageTensor(channels: Long): Boolean =
    size == 4 && (this[0] == -1L || this[0] == 1L) && this[1] == channels &&
        this[2] == ModelSpec.INPUT_SIZE.toLong() && this[3] == ModelSpec.INPUT_SIZE.toLong()
