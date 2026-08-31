# Android app

Native Kotlin/Jetpack Compose camera app for offline book-stack counting. The model is loaded once,
all frames stay on-device, and CameraX drops stale analysis frames instead of queueing them.

## Build

The checked-in Gradle daemon criteria select a standard Oracle JDK 21 installation and avoid the
GraalVM `jlink` incompatibility with Android 36's system modules. If Android Studio asks for a JDK,
select **Oracle OpenJDK 21** under **Settings → Build, Execution, Deployment → Build Tools →
Gradle → Gradle JDK**.

Then run:

```bash
cd apps/android
./gradlew testDebugUnitTest assembleDebug
```

If Gradle previously ran with GraalVM, stop that daemon once before rebuilding:

```bash
./gradlew --stop
./gradlew clean assembleDebug
```

The debug APK is written to `app/build/outputs/apk/debug/app-debug.apk`. The app module packages the
canonical files from `models/exported`; do not create another model copy under the Android source
tree. Both `separator_model.onnx` and its external data file must remain together.

## Architecture

```text
CameraX RGBA frame
  -> rotation + 640px letterbox + ImageNet normalization
  -> SeparatorModelBinding (ONNX Runtime)
  -> probability profiles + peak detection
  -> CountStabilizer
  -> neutral FrameResult
  -> Compose overlay and operator actions
```

`SeparatorModelBinding` is the only ONNX-aware application type. The rest of the app consumes
normalized, framework-neutral domain values. A future iOS adapter can expose the same values from
ONNX Runtime Objective-C/C++ without copying Android UI concepts.

The canonical color, spacing, radius, type, and motion values live in
`../design-system/tokens.json`; `ui/theme/Tokens.kt` is the Compose mapping.

## Device verification still required

The build and deterministic post-processing tests run on the host. Before production use, verify
camera orientation/preview alignment, model latency, memory, thermal behavior, and count parity on
representative physical devices and approved factory images.
