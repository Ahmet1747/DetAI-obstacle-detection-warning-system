package com.ahmetcan.yolov8tflite

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.IntentSender
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.BatteryManager
import android.os.Bundle
import android.os.Bundle as SpeechBundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.widget.Toast
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.ahmetcan.yolov8tflite.Constants.LABELS_PATH
import com.ahmetcan.yolov8tflite.Constants.MODEL_PATH
import com.ahmetcan.yolov8tflite.databinding.ActivityMainBinding
import com.google.android.gms.common.api.ResolvableApiException
import com.google.android.gms.location.*
import kotlinx.coroutines.*
import android.media.AudioManager
import android.media.ToneGenerator
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {

    private lateinit var binding: ActivityMainBinding
    private var detector: Detector? = null
    private lateinit var cameraExecutor: ExecutorService

    // ─── TTS Öncelik Yöneticisi ───────────────────────────────────────────────
    private lateinit var priorityManager: PriorityManager

    // ─── Nesne Tespiti Konum Throttle: sınıf → son seslendirildiği mesafe (birim) ──
    // Aynı nesne en az 100 birim (~10 cm) hareket etmedikçe tekrar seslenmez
    private val objectLastDistMap = mutableMapOf<String, Float>()

    // ─── Koltuk Doluluk Durumu Takibi: sınıf → son bilinen doluluk (true/false) ────
    // Doluluk değiştiğinde (boş→dolu veya dolu→boş) mesafe throttle bypass edilir
    private val seatLastOccupiedMap = mutableMapOf<String, Boolean>()

    // ─── YOLO Zaman Throttle: son uyarı zamanı (ms) ──────────────────────────────
    // Aynı anda en fazla 3000ms'de bir YOLO uyarısı → navigasyon sesini sürekli kesmez
    private var lastYoloWarningTimeMs = 0L
    private val YOLO_COOLDOWN_MS = 3_000L   // 3 saniye minimum aralık

    // Tarama modu
    private var isScanModeActive = false

    // GPS hazır mı? (0,0 koordinatıyla rota göndermeyi önler)
    private var isLocationReady = false

    // GPS uyarısı daha önce verildi mi? (bir kez uyar)
    private var gpsWarningSent = false

    // ─── Konum ────────────────────────────────────────────────────────────────
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var currentLat = 0.0
    private var currentLon = 0.0
    private var locationCallback: LocationCallback? = null

    // ─── Sesli Komut ──────────────────────────────────────────────────────────────────
    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    private var pendingVoiceListening = false  // Uzun basıldı → TTS bitince mikrofon açılacak

    // ─── Pil Durumu Takibi ──────────────────────────────────────────────────────────
    private var lowBatteryWarned = false  // %15 uyarısı bir kez verilsin
    private val batteryReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val level = intent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: return
            val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, 100)
            val percent = (level * 100) / scale

            if (percent <= 15 && !lowBatteryWarned) {
                lowBatteryWarned = true
                Log.d(TAG, "Pil uyarısı: %$percent")
                priorityManager.speakCriticalWarning(
                    "Pil seviyeniz yüzde $percent. Lütfen şarj edin. Uygulama yakında kapanabilir."
                )
            } else if (percent > 20) {
                // Pil tekrar %20’ın üstüne çıkarsa uyarıyı sıfırla (şarj edildiğinde)
                lowBatteryWarned = false
            }
        }
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val PREFS_NAME = "detai_prefs"
        private const val KEY_FIRST_LAUNCH = "is_first_launch"
    }

    // ─── onCreate ─────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        priorityManager = PriorityManager(this)

        // ─── TTS Hazır Olduğunda: Sadece İlk Kullanımda Sesli Rehber ────────────────
        priorityManager.onTtsReady = {
            val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            val isFirstLaunch = prefs.getBoolean(KEY_FIRST_LAUNCH, true)

            if (isFirstLaunch) {
                // İlk kullanım: detaylı sesli rehber (bir kerelik)
                prefs.edit().putBoolean(KEY_FIRST_LAUNCH, false).apply()
                priorityManager.speakNavigation(
                    "DetAI hazır. " +
                    "Ekrana çift dokunarak çevrenizi tarayabilirsiniz. " +
                    "Uzun basarak sesli komut verebilirsiniz. " +
                    "Navigasyon için, Taksim'e rota çiz, gibi komutlar söyleyebilirsiniz. " +
                    "Neredeyim diyerek konumunuzu öğrenebilirsiniz."
                )
                Log.d(TAG, "Onboarding: İlk kullanım rehberi seslendi")
            }
            // Sonraki açılışlarda sessiz başla
        }
        // TTS başladığında ses ikonunu göster
        priorityManager.onTtsStarted = {
            runOnUiThread {
                binding.ttsIndicator.visibility = android.view.View.VISIBLE
            }
        }

        // TTS bittiğinde: eğer kullanıcı uzun basmışsa mikrofonu aç, ayrıca ses ikonunu gizle
        priorityManager.onTtsDone = {
            runOnUiThread {
                binding.ttsIndicator.visibility = android.view.View.GONE
            }
            if (pendingVoiceListening) {
                pendingVoiceListening = false
                runOnUiThread {
                    lifecycleScope.launch {
                        delay(500L)  // TTS ses tamponunun bitmesini bekle
                        startVoiceListening()
                    }
                }
            }
        }

        val gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDoubleTap(e: MotionEvent): Boolean {
                if (!priorityManager.isSpeaking) {
                    isScanModeActive = true
                    priorityManager.speakCriticalWarning("Alan taranıyor, lütfen bekleyin.")
                }
                return true
            }

            override fun onLongPress(e: MotionEvent) {
                if (!priorityManager.isSpeaking && !isListening) {
                    Log.d(TAG, "Ekrana uzun basıldı → bip çalıp dinleme başlatılacak")
                    // TTS yerine kısa bip sesi çal (mikrofon TTS sesini duymasın)
                    try {
                        val toneGen = ToneGenerator(AudioManager.STREAM_NOTIFICATION, 80)
                        toneGen.startTone(ToneGenerator.TONE_PROP_BEEP, 200) // 200ms bip
                        toneGen.release()
                    } catch (e: Exception) {
                        Log.w(TAG, "Bip sesi çalınamadı: ${e.message}")
                    }
                    // Bip bittikten sonra mikrofonu aç
                    lifecycleScope.launch {
                        delay(400L) // Bip sesinin bitmesini bekle
                        startVoiceListening()
                    }
                }
            }
        })

        binding.root.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }

        // Kamera executor + dedektör
        cameraExecutor = Executors.newSingleThreadExecutor()
        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this) { toast(it) }
        }

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        if (allPermissionsGranted()) {
            startCamera()
            checkAndPromptLocationSettings()
            initSpeechRecognizer()
        } else {
            requestPermissionLauncher.launch(
                arrayOf(
                    Manifest.permission.CAMERA,
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.RECORD_AUDIO
                )
            )
        }

        // ─── Pil Durumu Dinleyicisi ─────────────────────────────────────────────────
        registerReceiver(batteryReceiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    }

    // ─── Nesne Tespit Callback ────────────────────────────────────────────────

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.setResults(boundingBoxes)
            binding.overlay.invalidate()

            if (boundingBoxes.isNotEmpty()) {
                if (isScanModeActive) scanEnvironment(boundingBoxes)
                else processSmartDetection(boundingBoxes)
            }
        }
    }

    // ─── Alan Taraması ────────────────────────────────────────────────────────

    private fun scanEnvironment(boxes: List<BoundingBox>) {
        isScanModeActive = false
        val sorted = boxes.sortedBy { it.distance }.take(3)
        val results = sorted.map { box ->
            val label = getSeatAwareLabel(box)
            "${box.positionText} ${formatDistance(box.distance)} mesafede $label"
        }
        val msg = "Etrafınızda şunlar var: ${results.joinToString(", ")}. Tarama tamamlandı."
        priorityManager.speakCriticalWarning(msg)
    }

    // ─── Akıllı Tespit — Risk Skorlaması (Tehlike / Mesafe) ────────

    private fun getDangerScore(clsName: String): Float {
        return when (clsName.lowercase()) {
            // Yüksek Tehlike (Kritik Engel/Araç)
            "car", "truck", "bus", "motorcycle", "stair", "crosswalk", "stop sign", "pole" -> 10f
            // Orta Tehlike (Dinamik Engel & Trafik Işığı)
            "person", "bicycle", "door", "traffic light" -> 5f
            // Düşük Tehlike (Zararsız/Zemin)
            "bench", "chair", "cat", "dog", "sidewalk" -> 2f
            else -> 1f
        }
    }

    private fun processSmartDetection(boxes: List<BoundingBox>) {
        // Eğer sistem şu anda bir YOLO uyarısı (örneğin "Önünüzde araba var") okuyorsa,
        // cümlenin yarım kesilmemesi için yeni gelen kamera karelerini tamamen görmezden gel ve konuşmanın bitmesini bekle.
        if (priorityManager.isSpeaking && !priorityManager.isSpeakingNavigation) {
            return
        }

        val now = System.currentTimeMillis()

        // ── Koltuk doluluk durumu güncelle (sadece state takibi) ────────────────
        val seatLabels = setOf("chair", "bench")
        val seats = boxes.filter { it.clsName.lowercase() in seatLabels }

        for (seat in seats) {
            if (seat.isOccupied == null) continue
            seatLastOccupiedMap[seat.clsName] = seat.isOccupied!!
        }

        // ── TÜM nesneleri (koltuklar dahil) risk skoruna göre sırala ──────────
        // En tehlikeli + en yakın nesne hangisiyse O söylenir
        val sortedByRisk = boxes.sortedByDescending { box ->
            val danger = getDangerScore(box.clsName)
            danger / java.lang.Math.max(1f, box.distance)
        }

        for (candidate in sortedByRisk) {
            val cls = candidate.clsName
            val isSeat = cls.lowercase() in seatLabels
            val throttleKey = if (isSeat) "seat_$cls" else cls

            // Konum bazlı throttle (Filtre 1)
            val lastDist = objectLastDistMap[throttleKey]
            if (lastDist != null &&
                Math.abs(candidate.distance - lastDist) < Constants.OBJECT_DISTANCE_THRESHOLD) {
                continue
            }

            // Zaman bazlı throttle (Filtre 2)
            if (now - lastYoloWarningTimeMs < YOLO_COOLDOWN_MS) {
                Log.v(TAG, "YOLO cooldown: yeni uyarı atlandı")
                return
            }

            // Filtrelere takılmayan en yüksek riskli nesneyi bulduk
            lastYoloWarningTimeMs = now
            objectLastDistMap[throttleKey] = candidate.distance

            val name = getSeatAwareLabel(candidate)
            val dist = formatDistance(candidate.distance)
            val dir  = candidate.positionText
            val msg  = "$dir $dist mesafede $name var."
            Log.d(TAG, "YOLO uyarı: $cls dist=${candidate.distance} → '$msg'")

            priorityManager.speakCriticalWarning(msg)
            return // En riskli 1 nesneyi söyledik, bu frame için yeterli
        }
    }

    // ─── Konum Güncellemeleri + Manevra Kontrolü ─────────────────────────────

    // GPS kapalıysa açması için sistem diyaloğu gösterilir
    private val locationSettingsLauncher =
        registerForActivityResult(ActivityResultContracts.StartIntentSenderForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                Log.d(TAG, "GPS kullanıcı tarafından açıldı.")
                startLocationUpdates()
            } else {
                Log.w(TAG, "GPS açılmadı — konum devre dışı.")
                toast("İyi navigasyon için Konum'u açın.")
            }
        }

    /** GPS kapalıysa sistem diyaloğu göster, açıksa doğrudan konum güncellemesini başlat. */
    private fun checkAndPromptLocationSettings() {
        val request = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY, 5_000L
        ).setMinUpdateIntervalMillis(3_000L).build()

        val settingsRequest = LocationSettingsRequest.Builder()
            .addLocationRequest(request)
            .setAlwaysShow(true)
            .build()

        LocationServices.getSettingsClient(this)
            .checkLocationSettings(settingsRequest)
            .addOnSuccessListener {
                Log.d(TAG, "GPS zaten açık, konum güncellemeleri başlatılıyor.")
                startLocationUpdates()
            }
            .addOnFailureListener { exception ->
                if (exception is ResolvableApiException) {
                    try {
                        val intentSenderRequest = IntentSenderRequest.Builder(
                            exception.resolution.intentSender
                        ).build()
                        locationSettingsLauncher.launch(intentSenderRequest)
                    } catch (e: IntentSender.SendIntentException) {
                        Log.e(TAG, "GPS diyaloğu gösterilemedi: ${e.message}")
                    }
                } else {
                    Log.e(TAG, "GPS ayarları çözümlenemedi: ${exception.message}")
                }
            }
    }

    @SuppressLint("MissingPermission")
    private fun startLocationUpdates() {
        val request = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY, 5_000L
        ).setMinUpdateIntervalMillis(3_000L).build()

        locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation
                if (loc == null) {
                    Log.d("DetAI_Debug", "GPS: onLocationResult çağrıldı ama lastLocation NULL")
                    return
                }
                currentLat = loc.latitude
                currentLon = loc.longitude

                // Kullanıcının baktığı yönü navigasyona aktar (göreceli yön hesabı için)
                if (loc.hasBearing()) {
                    NavigationService.currentUserBearing = loc.bearing
                }

                if (!isLocationReady) {
                    isLocationReady = true
                    gpsWarningSent = false   // GPS geldi — bir sonraki kopuşta tekrar uyarabilir
                    Log.d("DetAI_Debug", "GPS: İLK FIX alındı → lat=$currentLat lon=$currentLon bearing=${loc.bearing}° accuracy=${loc.accuracy}m")
                } else {
                    Log.d("DetAI_Debug", "GPS: güncelleme → $currentLat,$currentLon bearing=${loc.bearing}° acc=${loc.accuracy}m")
                }

                // Her GPS güncellemesinde HTTP çağrısı YOK — sadece yerel Haversine kontrolü
                val maneuver = NavigationService.checkManeuver(currentLat, currentLon)
                if (maneuver != null) {
                    Log.d("DetAI_Debug", "GPS: Manevra tetiklendi → '${maneuver.text}' (${maneuver.distanceMeters.toInt()}m)")
                    priorityManager.speakNavigation(maneuver.text)
                }
            }
        }

        // Son bilinen konumu hemen al (GPS güncellenmesini bekleme)
        fusedLocationClient.lastLocation.addOnSuccessListener { loc ->
            if (loc != null) {
                currentLat = loc.latitude
                currentLon = loc.longitude
                isLocationReady = true
                Log.d(TAG, "Son bilinen konum alındı: $currentLat, $currentLon")
            }
        }

        fusedLocationClient.requestLocationUpdates(request, locationCallback!!, mainLooper)
    }

    // ─── Sesli Komut — SpeechRecognizer ──────────────────────────────────────

    private fun initSpeechRecognizer() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Log.w(TAG, "Konuşma tanıma bu cihazda desteklenmiyor.")
            return
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: SpeechBundle?) {
                isListening = true
                Log.d(TAG, "Sesli komut bekleniyor...")
            }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() { isListening = false }

            override fun onResults(results: SpeechBundle?) {
                isListening = false
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                matches?.firstOrNull()?.let { handleVoiceCommand(it.lowercase()) }
                // Mikrofon kapanır — tekrar uzun basana kadar açılmaz
                Log.d(TAG, "Sesli komut alındı, mikrofon kapatıldı")
            }

            override fun onError(error: Int) {
                isListening = false
                val errMsg = when (error) {
                    SpeechRecognizer.ERROR_AUDIO            -> "Ses kaydı hatası"
                    SpeechRecognizer.ERROR_CLIENT           -> "İstemci hatası"
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Mikrofon izni yok!"
                    SpeechRecognizer.ERROR_NETWORK          -> "Ağ hatası"
                    SpeechRecognizer.ERROR_NETWORK_TIMEOUT  -> "Ağ zaman aşımı"
                    SpeechRecognizer.ERROR_NO_MATCH         -> "Ses eşleşmedi"
                    SpeechRecognizer.ERROR_RECOGNIZER_BUSY  -> "Tanıyıcı meşgul"
                    SpeechRecognizer.ERROR_SERVER           -> "Sunucu hatası"
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT   -> "Konuşma zaman aşımı"
                    else -> "Bilinmeyen hata ($error)"
                }
                Log.w(TAG, "SpeechRecognizer onError: $errMsg")

                // Kullanıcıya sadece önemli hataları söyle
                if (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                    // Ses algılanamadı — sessizce kapat
                    Log.d(TAG, "Ses algılanamadı, mikrofon kapatıldı. Tekrar denemek için uzun basın.")
                } else if (error == SpeechRecognizer.ERROR_CLIENT || error == SpeechRecognizer.ERROR_SERVER) {
                    // Kritik hata — SpeechRecognizer’ı yeniden yarat
                    Log.w(TAG, "Kritik hata ($error), SpeechRecognizer yeniden başlatılıyor...")
                    speechRecognizer?.destroy()
                    speechRecognizer = null
                    lifecycleScope.launch {
                        delay(2000)
                        initSpeechRecognizer()
                    }
                } else {
                    toast("🎙️ Hata: $errMsg")
                }
            }

            override fun onPartialResults(partialResults: SpeechBundle?) {}
            override fun onEvent(eventType: Int, params: SpeechBundle?) {}
        })
        // Mikrofon otomatik açılmaz — sadece uzun basınca başlar
        Log.d(TAG, "SpeechRecognizer hazır. Mikrofon için ekrana uzun basın.")
    }

    /** Mikrofonu aç (sadece uzun basma sonrası çağrılır) */
    private fun startVoiceListening() {
        if (isListening) return
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "tr-TR")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        try {
            speechRecognizer?.startListening(intent)
            Log.d(TAG, "Mikrofon açıldı — kullanıcı konuşabilir")
        } catch (e: Exception) {
            Log.e(TAG, "Dinleme başlatılamadı: ${e.message}")
        }
    }

    // ─── Sesli Komut İşleme ───────────────────────────────────────────────────

    /**
     * Desteklenen komutlar (Türkçe doğal dil):
     *   "[yer adı]'na rota çiz"        → Geocoding → rota indir
     *   "[yer adı]'e git"              → Geocoding → rota indir
     *   "[yer adı]'ye gidiyorum"       → Geocoding → rota indir
     *   "navigasyon başlat [yer adı]"  → Geocoding → rota indir
     *   "navigasyonu durdur"           → rotayı temizle
     *   "rotayı iptal et"              → rotayı temizle
     */
    private fun handleVoiceCommand(command: String) {
        Log.d(TAG, "Sesli komut: $command")
        toast("🗣️ Duyulan: $command") // Kullanıcı komutun algılanıp algılanmadığını görsün

        // ─ Durdurma komutları (önce kontrol et, çakışmayı önler) ─────────────
        val isStopCommand = command.contains("durdur") || command.contains("iptal") ||
                            command.contains("birak") || command.contains("dur")
        if (isStopCommand && !command.contains("rota çiz")) {
            NavigationService.clearRoute()
            priorityManager.speakNavigation("Navigasyon durduruldu.")
            return
        }

        // ─ Konum Sorma Komutları ─────────────────────────────────────────────
        val isLocationQuery = command.contains("neredeyim") || command.contains("konumum") ||
                              command.contains("hangi sokak") || command.contains("hangi mahalle") || 
                              command.contains("burası neresi")
        if (isLocationQuery) {
            if (!isLocationReady) {
                priorityManager.speakCriticalWarning("GPS sinyali bekleniyor, şu anki konumunuzu bulamıyorum.")
                return
            }

            // Aktif rota varsa önce rota durumunu söyle
            val routeStatus = NavigationService.getRouteStatus(currentLat, currentLon)
            if (routeStatus != null) {
                priorityManager.speakNavigation(routeStatus)
                return
            }

            priorityManager.speakNavigation("Konumunuz bulunuyor...")
            lifecycleScope.launch {
                val addressName = NavigationService.getCurrentAddressName(currentLat, currentLon)
                if (addressName != null) {
                    priorityManager.speakNavigation("Şu anda $addressName civarındasınız.")
                } else {
                    priorityManager.speakCriticalWarning("Şu anki adresiniz tespit edilemedi.")
                }
            }
            return
        }

        // ─ Başlatma komutları — hedef ayıklama ───────────────────────────────
        // Strateji: komutun sonundan tetikleyici kelimeyi bul, öncesini yer adı olarak al.
        // Sondaki Türkçe yapı eklerini yer adından çıkar (stripSuffix kullanır).

        // Sondaki eki kaldırıp yer adını temizle
        fun stripSuffix(raw: String): String {
            val r = raw.trim().trimEnd()
            // Önce apostrof ile ayrılmış eki kontrol et
            for (s in listOf("'ye","'ya","'ne","'na","'te","'ta","'de","'da",
                             "\u2019ye","\u2019ya","\u2019ne","\u2019na")) {
                if (r.endsWith(s)) return r.dropLast(s.length)
            }
            // Apostrof yoksa Turkish vowel harmony eklerini kontrol et
            // Sadece gerçekten ek ise kaldır (minimum 3 karakter kalsın)
            for (s in listOf("ye","ya","ne","na","te","ta","de","da")) {
                if (r.length > s.length + 2 && r.endsWith(s)) {
                    return r.dropLast(s.length)
                }
            }
            for (s in listOf("e","a")) {
                if (r.length > s.length + 2 && r.endsWith(s)) {
                    val beforeSuffix = r[r.length - s.length - 1]
                    if (!beforeSuffix.lowercaseChar().let { it == 'a'||it=='e'||it=='i'||it=='ı'||it=='o'||it=='ö'||it=='u'||it=='ü' }) {
                        // Eğer harften önce kesme işareti de varsa (örn. nevşehir'e -> nevşehir)
                        var result = r.dropLast(s.length)
                        if (result.endsWith("'") || result.endsWith("\u2019")) {
                            result = result.dropLast(1)
                        }
                        return result
                    }
                }
            }
            return r
        }

        val triggerKeywords = listOf(
            "rota çiz",
            "git",
            "gidiyorum",
            "yönlendir",
            "yol tarifi",
            "navigasyon başlat",
            "navigasyon"
        )

        var destination: String? = null

        // "önce spesifik kalıp" → rota çiz X veya navigasyon başlat X
        val prefixPatterns = listOf(
            Regex("""rota\s+çiz\s+(.+)"""),
            Regex("""navigasyon\s+başlat\s+(.+)""")
        )
        for (pattern in prefixPatterns) {
            val match = pattern.find(command)
            if (match != null) {
                destination = match.groupValues[1].trim().takeIf { it.isNotBlank() }
                if (destination != null) break
            }
        }

        // Suffix kalıbı: X + [ek] + tetikleyici
        if (destination == null) {
            val escapedTriggers = triggerKeywords.joinToString("|") { Regex.escape(it) }
            // Greedy: yer adı + isteğe bağlı ek + boşluk + tetikleyici
            val pattern = Regex("""^(.+?)\s+(?:$escapedTriggers)""")
            val match = pattern.find(command)
            if (match != null) {
                destination = stripSuffix(match.groupValues[1].trim()).takeIf { it.isNotBlank() }
            }
        }

        // Eğer bir hedef kelimesi ayıklandıysa → geocoding + rota
        if (destination != null) {
            val dest = destination

            // GPS henüz hazır değilse HER SEFERİNDE uyar (önceden sadece 1 kez uyarıyordu,
            // kullanıcı tekrar tekrar deneyip sessizce reddediliyordu)
            if (!isLocationReady) {
                Log.d("DetAI_Debug", "fetchRoute → GPS hazır değil, istek iptal edildi")
                priorityManager.speakCriticalWarning("GPS sinyali bekleniyor. Lütfen açık bir alanda bekleyin.")
                return
            }

            priorityManager.speakNavigation("$dest aranıyor.")
            lifecycleScope.launch {
                // 1) Geocoding (IO thread)
                val coords = NavigationService.geocodePlace(dest)
                if (coords == null) {
                    priorityManager.speakCriticalWarning(
                        "$dest bulunamadı. Lütfen daha belirgin bir yer adı söyleyin."
                    )
                    return@launch
                }
                val (destLat, destLon) = coords
                val resolvedName = NavigationService.lastDestinationName

                priorityManager.speakNavigation("$resolvedName bulundu. Rota hesaplanıyor.")

                // 2) Rota indir (IO thread)
                Log.d("DetAI_Debug", "fetchRoute → başlatılıyor: $resolvedName ($destLat,$destLon) ← ($currentLat,$currentLon)")
                val success = NavigationService.fetchRoute(
                    startLat = currentLat,
                    startLon = currentLon,
                    destLat  = destLat,
                    destLon  = destLon
                )

                if (success) {
                    val totalKm = "%.1f".format(NavigationService.totalDistanceM / 1000)
                    val totalMin = NavigationService.totalDurationS / 60
                    Log.d("DetAI_Debug", "fetchRoute → SUCCESS: $resolvedName, $totalKm km, $totalMin dk")

                    val navMsg = "Rota hazır. $resolvedName'a yaklaşık $totalKm kilometre, " +
                                 "yürüyüşle tahminen $totalMin dakika."
                    priorityManager.speakNavigation(navMsg)

                    // İlk 3 adımın özetini söyle
                    val firstInstruction = NavigationService.firstInstructionText
                    if (firstInstruction.isNotBlank() && firstInstruction != navMsg) {
                        Log.d("DetAI_Debug", "fetchRoute → Rota özeti: $firstInstruction")
                        priorityManager.speakNavigation(firstInstruction)
                    }
                } else {
                    Log.d("DetAI_Debug", "fetchRoute → ERROR: Rota hesaplanamadı ($resolvedName)")
                    priorityManager.speakCriticalWarning(
                        "Rota hesaplanamadı. İnternet bağlantınızı kontrol edin."
                    )
                }
            }
        } else if (command.contains("rota") || command.contains("navigasyon başlat") ||
                   command.contains("yola çık")) {
            // Hedef yok ama rota komutu var → kullanıcıya sor
            priorityManager.speakNavigation(
                "Nereye gitmek istiyorsunuz? Örneğin: Taksim'e rota çiz."
            )
        }
    }


    // ─── Kamera ───────────────────────────────────────────────────────────────

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build()
                .also { it.surfaceProvider = binding.viewFinder.surfaceProvider }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { image ->
                        val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                        image.use { bitmap.copyPixelsFromBuffer(image.planes[0].buffer) }
                        val matrix = Matrix().apply {
                            postRotate(image.imageInfo.rotationDegrees.toFloat())
                        }
                        val rotatedBitmap = Bitmap.createBitmap(
                            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                        )
                        detector?.detect(rotatedBitmap)
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer
                )
                // Kamera bağlandıktan sonra gerçek odak uzaklığını hesapla
                computeAndSetFocalLength()
            } catch (exc: Exception) {
                Log.e(TAG, "Kamera başlatılamadı", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Kamera sensöründen gerçek odak uzaklığını piksel cinsinden hesaplar.
     *
     * Formül: focalLengthPx = focalLengthMm × (tensorWidth / sensorWidthMm)
     *
     * Bu sayede her cihazda mesafe hesabı doğru çalışır;
     * sabit 500f değeri yerine gerçek kamera parametreleri kullanılır.
     */
    private fun computeAndSetFocalLength() {
        try {
            val cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
            // Arka kamerayı bul (genelde "0" ID)
            val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
                val chars = cameraManager.getCameraCharacteristics(id)
                chars.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            } ?: cameraManager.cameraIdList.firstOrNull() ?: return

            val chars = cameraManager.getCameraCharacteristics(cameraId)

            // Odak uzaklığı (mm)
            val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val focalLengthMm = focalLengths?.firstOrNull() ?: return

            // Sensör fiziksel boyutu (mm)
            val sensorSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return
            val sensorWidthMm = sensorSize.width

            if (sensorWidthMm <= 0f) return

            // Model giriş tensoru genişliği (piksel) — örn. 640
            val inputShape = (detector?.tensorWidth ?: 640).toFloat()

            val focalLengthPx = focalLengthMm * (inputShape / sensorWidthMm)

            detector?.focalLengthPx = focalLengthPx

            Log.d(TAG, "Dinamik odak uzaklığı hesaplandı: " +
                    "focalMm=$focalLengthMm, sensorW=${sensorWidthMm}mm, " +
                    "tensorW=$inputShape → focalPx=$focalLengthPx")
        } catch (e: Exception) {
            Log.w(TAG, "Odak uzaklığı hesaplanamadı, varsayılan 500f kullanılacak: ${e.message}")
        }
    }

    // ─── İzin Yönetimi ────────────────────────────────────────────────────────

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { perms ->
            if (perms[Manifest.permission.CAMERA] == true) startCamera()
            if (perms[Manifest.permission.ACCESS_FINE_LOCATION] == true) checkAndPromptLocationSettings()
            if (perms[Manifest.permission.RECORD_AUDIO] == true) initSpeechRecognizer()
        }

    private fun allPermissionsGranted(): Boolean {
        val required = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.RECORD_AUDIO
        )
        return required.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    // ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────

    private fun formatDistance(dist: Float): String {
        return if (dist >= 100f) {
            val m  = (dist / 100).toInt()
            val cm = (dist % 100).toInt()
            if (cm > 0) "$m metre $cm santimetre" else "$m metre"
        } else {
            "${dist.toInt()} santimetre"
        }
    }

    /**
     * Koltuk / Sandalye doluluk bilgisiyle birlikte Türkçe etiket döndürür.
     *
     * chair → "boş sandalye" veya "dolu sandalye"
     * bench → "boş bank"     veya "dolu bank"
     * Diğer sınıflar → normal translateLabel çıktısı (kişi, araba, vb.)
     */
    private fun getSeatAwareLabel(box: BoundingBox): String {
        val baseName = translateLabel(box.clsName)
        return when (box.isOccupied) {
            false -> "boş $baseName"
            true  -> "dolu $baseName"
            null  -> baseName  // Oturulabilir nesne değil, normal etiket
        }
    }

    private fun translateLabel(label: String): String {
        return when (label.lowercase()) {
            "bench"         -> "bank"
            "bicycle"       -> "bisiklet"
            "bus"           -> "otobüs"
            "car"           -> "araba"
            "cat"           -> "kedi"
            "chair"         -> "sandalye"
            "crosswalk"     -> "yaya geçidi"
            "dog"           -> "köpek"
            "door"          -> "kapı"
            "motorcycle"    -> "motor"
            "person"        -> "kişi"
            "pole"          -> "direk"
            "sidewalk"      -> "kaldırım"
            "stair"         -> "merdiven"
            "stop sign"     -> "dur tabelası"
            "traffic light" -> "trafik ışığı"
            "truck"         -> "kamyon"
            else            -> label
        }
    }

    private fun toast(message: String) = runOnUiThread {
        Toast.makeText(baseContext, message, Toast.LENGTH_SHORT).show()
    }

    override fun onEmptyDetect() = runOnUiThread { binding.overlay.clear() }

    // ─── Lifecycle ────────────────────────────────────────────────────────────

    override fun onResume() {
        super.onResume()
    }

    override fun onPause() {
        super.onPause()
        // Arka plana geçince dinlemeyi durdur
        speechRecognizer?.stopListening()
        isListening = false
    }

    override fun onDestroy() {
        super.onDestroy()
        try { unregisterReceiver(batteryReceiver) } catch (_: Exception) {}
        locationCallback?.let { fusedLocationClient.removeLocationUpdates(it) }
        speechRecognizer?.destroy()
        priorityManager.shutdown()
        detector?.close()
        cameraExecutor.shutdown()
    }
}