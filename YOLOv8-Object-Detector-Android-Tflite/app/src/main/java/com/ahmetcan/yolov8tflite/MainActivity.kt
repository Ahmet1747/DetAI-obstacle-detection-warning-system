package com.ahmetcan.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.ahmetcan.yolov8tflite.Constants.LABELS_PATH
import com.ahmetcan.yolov8tflite.Constants.MODEL_PATH
import com.ahmetcan.yolov8tflite.databinding.ActivityMainBinding
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private var detector: Detector? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var textToSpeech: TextToSpeech

    // Akıllı Takip ve Durum Değişkenleri
    private var lastSpokenObjectName = ""
    private var lastSpokenDistance = 0f
    private var lastSpeakTime = 0L
    private var isScanModeActive = false

    // YENİ: Ses motorunun konuşup konuşmadığını takip eder
    private var isSpeaking = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // TTS Ayarları ve Takılma Önleyici Listener
        textToSpeech = TextToSpeech(this) { status ->
            if (status != TextToSpeech.ERROR) {
                textToSpeech.language = Locale("tr", "TR")
                textToSpeech.setSpeechRate(1.1f) // Daha net anlaşılması için hafif yavaşlatıldı
                textToSpeech.setPitch(1.0f)

                // Konuşma durumunu izleyen dinleyici
                textToSpeech.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) { isSpeaking = true }
                    override fun onDone(utteranceId: String?) { isSpeaking = false }
                    override fun onError(utteranceId: String?) { isSpeaking = false }
                })
            }
        }

        // TARA BUTONU: Sadece butona basıldığında alan taranır
        binding.btnScan.setOnClickListener {
            if (!isSpeaking) {
                isScanModeActive = true
                speak("Alan taranıyor, lütfen bekleyin.", "SCAN_START")
            }
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this) { toast(it) }
        }

        if (allPermissionsGranted()) startCamera()
        else requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.setResults(boundingBoxes)
            binding.overlay.invalidate()

            if (boundingBoxes.isNotEmpty()) {
                if (isScanModeActive) {
                    scanEnvironment(boundingBoxes)
                } else {
                    // Cihaz o an konuşmuyorsa yeni tespiti işle
                    if (!isSpeaking) {
                        processSmartDetection(boundingBoxes)
                    }
                }
            }
        }
    }

    private fun scanEnvironment(boxes: List<BoundingBox>) {
        isScanModeActive = false

        val sortedBoxes = boxes.sortedBy { it.distance }.take(3)
        val scanResults = mutableListOf<String>()

        for (box in sortedBoxes) {
            val name = translateLabel(box.clsName)
            scanResults.add("${box.positionText} $name")
        }

        val finalMessage = "Etrafınızda şunlar var: " + scanResults.joinToString(", ") + ". Tarama tamamlandı."
        speak(finalMessage, "SCAN_RESULT")
        lastSpeakTime = System.currentTimeMillis()
    }

    private fun processSmartDetection(boxes: List<BoundingBox>) {
        val currentTime = System.currentTimeMillis()
        val nearestBox = boxes.minByOrNull { it.distance } ?: return

        val currentObjectName = nearestBox.clsName
        val currentDistance = nearestBox.distance

        val isDifferentObject = currentObjectName != lastSpokenObjectName

        // HATA ÖNLEYİCİ: Mesafe eşiği 25cm'ye çıkarıldı (el titremesi kaynaklı takılmaları önler)
        val hasSignificantMovement = Math.abs(currentDistance - lastSpokenDistance) >= 25f

        // HATA ÖNLEYİCİ: Konuşmalar arası mutlak 4 saniye bekleme süresi
        val waitPeriod = 4000

        if (isDifferentObject || (hasSignificantMovement && currentTime - lastSpeakTime > waitPeriod)) {
            val turkishName = translateLabel(currentObjectName)
            val distanceText = formatDistance(currentDistance)
            val direction = nearestBox.positionText

            val message = "$direction, $distanceText mesafede $turkishName var."

            // QUEUE_FLUSH: Mevcut kuyruğu temizle ve yeni cümleye başla
            speak(message, currentObjectName)

            lastSpokenObjectName = currentObjectName
            lastSpokenDistance = currentDistance
            lastSpeakTime = currentTime
        }
    }

    // Merkezi seslendirme fonksiyonu
    private fun speak(text: String, id: String) {
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
    }

    private fun formatDistance(dist: Float): String {
        return if (dist >= 100f) {
            val meters = (dist / 100).toInt()
            val cm = (dist % 100).toInt()
            if (cm > 0) "$meters metre $cm santimetre" else "$meters metre"
        } else {
            "${dist.toInt()} santimetre"
        }
    }

    private fun translateLabel(label: String): String {
        return when (label.lowercase()) {
            "person" -> "Kişi"
            "bicycle" -> "Bisiklet"
            "motorcycle" -> "Motosiklet"
            "car" -> "Araba"
            "bus" -> "Otobüs"
            "truck" -> "Kamyon"
            "dog" -> "Köpek"
            "cat" -> "Kedi"
            "bottle" -> "Şişe"
            "chair" -> "Sandalye"
            "bench" -> "Bank"
            "couch" -> "Kanepe"
            "potted plant" -> "Saksı bitkisi"
            "bed" -> "Yatak"
            "dining table" -> "Yemek masası"
            "traffic light" -> "Trafik ışığı"
            "fire hydrant" -> "Yangın musluğu"
            "stop sign" -> "Dur tabelası"
            "parking meter" -> "Park metresi"
            "tv" -> "Televizyon"
            "laptop" -> "Laptop"
            "cell phone" -> "Cep telefonu"
            "microwave" -> "Mikrodalga"
            "oven" -> "Fırın"
            "toaster" -> "Ekmek kızartma makinesi"
            "sink" -> "Lavabo"
            "refrigerator" -> "Buzdolabı"
            "book" -> "Kitap"
            "clock" -> "Saat"
            "vase" -> "Vazo"
            "umbrella" -> "Şemsiye"
            "bird" -> "Kuş"
            "mouse" -> "Mouse"
            else -> label
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also { it.surfaceProvider = binding.viewFinder.surfaceProvider }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                        image.use { bitmap.copyPixelsFromBuffer(image.planes[0].buffer) }
                        val matrix = Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }
                        val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                        detector?.detect(rotatedBitmap)
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("MainActivity", "Kamera başlatılamadı", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
        if (permissions[Manifest.permission.CAMERA] == true) startCamera()
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun toast(message: String) = runOnUiThread {
        Toast.makeText(baseContext, message, Toast.LENGTH_SHORT).show()
    }

    override fun onEmptyDetect() = runOnUiThread { binding.overlay.clear() }

    override fun onDestroy() {
        super.onDestroy()
        if (::textToSpeech.isInitialized) {
            textToSpeech.stop()
            textToSpeech.shutdown()
        }
        detector?.close()
        cameraExecutor.shutdown()
    }
}