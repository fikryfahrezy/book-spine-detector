package com.bookspine.detector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.bookspine.detector.scan.CameraPreview
import com.bookspine.detector.scan.ScanUi
import com.bookspine.detector.scan.ScanViewModel
import com.bookspine.detector.ui.theme.AppColors
import com.bookspine.detector.ui.theme.BookSpineTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            BookSpineTheme {
                BookSpineApp()
            }
        }
    }
}

@Composable
private fun BookSpineApp(scanViewModel: ScanViewModel = viewModel()) {
    val context = androidx.compose.ui.platform.LocalContext.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED,
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> hasCameraPermission = granted }
    LaunchedEffect(Unit) {
        if (!hasCameraPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    val state by scanViewModel.state.collectAsStateWithLifecycle()
    Box(Modifier.fillMaxSize().background(AppColors.Ink)) {
        if (hasCameraPermission) {
            CameraPreview(scanViewModel, Modifier.fillMaxSize())
        }
        ScanUi(
            state = state,
            hasCameraPermission = hasCameraPermission,
            onRequestPermission = { permissionLauncher.launch(Manifest.permission.CAMERA) },
            onTogglePause = scanViewModel::togglePause,
            onRescan = scanViewModel::rescan,
            onConfirm = scanViewModel::confirm,
            onHumanReview = scanViewModel::requestHumanReview,
        )
    }
}
