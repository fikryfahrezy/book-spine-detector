package com.bookspine.detector.inference

import android.content.Context
import java.io.File

internal object ModelFiles {
    fun install(context: Context): File {
        val directory = File(context.noBackupFilesDir, "models/v1").apply { mkdirs() }
        val model = copyAssetIfNeeded(context, ModelSpec.MODEL_FILE, directory)
        copyAssetIfNeeded(context, ModelSpec.EXTERNAL_DATA_FILE, directory)
        return model
    }

    private fun copyAssetIfNeeded(context: Context, name: String, directory: File): File {
        val destination = File(directory, name)
        context.assets.openFd(name).use { descriptor ->
            if (destination.exists() && destination.length() == descriptor.length) {
                return destination
            }
        }

        val temporary = File(directory, "$name.partial")
        context.assets.open(name).use { input ->
            temporary.outputStream().buffered().use(input::copyTo)
        }
        if (destination.exists()) check(destination.delete()) { "Could not replace model artifact $name" }
        check(temporary.renameTo(destination)) { "Could not install model artifact $name" }
        return destination
    }
}
