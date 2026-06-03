package com.ahmetcan.yolov8tflite

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.UUID

/**
 * PriorityManager — Tüm TTS çıktısını öncelik kurallarıyla yönetir.
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  KURAL 1: YOLO uyarısı → tts.stop() → anında seslendir        │
 * │  KURAL 2: Uyarı bitince → kesilen nav talimatı otomatik devam  │
 * │  KURAL 3: Nav talimatı QUEUE_FLUSH ile hemen söylenir          │
 * └─────────────────────────────────────────────────────────────────┘
 */
class PriorityManager(context: Context) {

    private val TAG = "PriorityManager"

    // ─── Durum ────────────────────────────────────────────────────────────────
    var isSpeaking: Boolean = false
        private set

    var isSpeakingNavigation: Boolean = false
        private set

    // YOLO tarafından kesilen navigasyon talimatı (otomatik resume için)
    @Volatile
    private var pendingNavText: String? = null

    // Şu an konuşulan navigasyon metni (YOLO gelince kesmek için)
    @Volatile
    private var currentNavText: String? = null

    private var tts: TextToSpeech? = null
    private var isReady = false

    // TTS başladığında mikrofonu kapat
    var onTtsStarted: (() -> Unit)? = null
    // TTS tamamen bitti → mikrofonu yeniden aç
    var onTtsDone: (() -> Unit)? = null
    // TTS motoru hazır olduğunda çağrılır (onboarding için)
    var onTtsReady: (() -> Unit)? = null

    // ─── Başlatma ─────────────────────────────────────────────────────────────
    init {
        tts = TextToSpeech(context, { status ->
            if (status == TextToSpeech.SUCCESS) {
                val langResult = tts?.setLanguage(Locale("tr", "TR"))
                val langStatus = when (langResult) {
                    TextToSpeech.LANG_AVAILABLE             -> "AVAILABLE ✓"
                    TextToSpeech.LANG_COUNTRY_AVAILABLE     -> "COUNTRY_AVAILABLE ✓"
                    TextToSpeech.LANG_COUNTRY_VAR_AVAILABLE -> "COUNTRY_VAR_AVAILABLE ✓"
                    TextToSpeech.LANG_MISSING_DATA          -> "MISSING_DATA ✗"
                    TextToSpeech.LANG_NOT_SUPPORTED         -> "NOT_SUPPORTED ✗"
                    else                                    -> "UNKNOWN($langResult)"
                }
                if (langResult == TextToSpeech.LANG_MISSING_DATA ||
                    langResult == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "TTS: Türkçe dil paketi yok ($langStatus)")
                    tts?.language = Locale.getDefault()
                } else {
                    Log.d(TAG, "TTS: Türkçe dil kurulumu başarılı ($langStatus)")
                }

                tts?.setSpeechRate(1.05f)
                tts?.setPitch(1.0f)

                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        isSpeaking = true
                        isSpeakingNavigation = utteranceId?.startsWith("NAV_") == true
                        if (isSpeakingNavigation) {
                            Log.d(TAG, "NAV başladı → mikrofon kapatılıyor")
                        }
                        onTtsStarted?.invoke()
                    }

                    override fun onDone(utteranceId: String?) {
                        isSpeaking = false
                        
                        if (utteranceId?.startsWith("NAV_") == true) {
                            isSpeakingNavigation = false
                            currentNavText = null // Navigasyon başarıyla tamamlandı
                        } else {
                            isSpeakingNavigation = false
                        }

                        // Kritik uyarı bitti → kesilen navigasyonu devam ettir
                        if (utteranceId?.startsWith("CRITICAL_") == true) {
                            val pending = pendingNavText
                            if (!pending.isNullOrBlank()) {
                                pendingNavText = null
                                Log.d(TAG, "YOLO bitti → Nav resume: \"${pending.take(60)}\"")
                                // Resume etmeden önce kısa bekle (ses akustik gürültüsünü önler)
                                Thread.sleep(300)
                                speakNavigation(pending)
                                return  // onTtsDone'u tetikleme, nav devam edecek
                            }
                        }
                        onTtsDone?.invoke()
                    }

                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) {
                        isSpeaking = false
                        if (utteranceId?.startsWith("NAV_") == true) {
                            isSpeakingNavigation = false
                            currentNavText = null
                        } else {
                            isSpeakingNavigation = false
                        }
                        onTtsDone?.invoke()
                    }
                    
                    override fun onStop(utteranceId: String?, interrupted: Boolean) {
                        super.onStop(utteranceId, interrupted)
                        isSpeaking = false
                        if (utteranceId?.startsWith("NAV_") == true) {
                            isSpeakingNavigation = false
                            // currentNavText'i BURADA temizlemiyoruz çünkü kesildi (interrupted).
                            // speakCriticalWarning içinde pendingNavText'e aktarılmış olacak.
                        } else {
                            isSpeakingNavigation = false
                        }
                    }
                })

                isReady = true
                Log.d(TAG, "TTS hazır. Motor=${tts?.defaultEngine}")

                // MainActivity'ye TTS hazır olduğunu bildir (onboarding için)
                onTtsReady?.invoke()
            } else {
                Log.e(TAG, "TTS başlatılamadı. Status: $status")
            }
        })
    }

    // ─── Kritik Uyarı (YOLO nesne tespiti) ───────────────────────────────────

    /**
     * YOLO nesne uyarısı.
     * - Navigasyon konuşuyorsa metni saklar (resume için)
     * - tts.stop() ile anında keser
     * - Uyarıyı söyler
     * - Uyarı bitince [onDone] içinde navigasyon otomatik devam eder
     */
    fun speakCriticalWarning(text: String) {
        if (!isReady) return

        // Navigasyon kuyrukta veya konuşuluyorsa metni sakla
        val nav = currentNavText
        if (!nav.isNullOrBlank()) {
            pendingNavText = nav
            Log.d(TAG, "YOLO: Navigasyon kesildi/kuyruktaydı, resume kaydedildi: \"${nav.take(60)}\"")
            currentNavText = null // Aynı metni tekrar tekrar kaydetmemek için temizle
        }

        tts?.stop()
        isSpeaking = false
        isSpeakingNavigation = false

        val id = "CRITICAL_${UUID.randomUUID()}"
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
        Log.d(TAG, "KRİTİK UYARI: ${text.take(80)}")
    }

    /**
     * Dışarıdan (MainActivity) resume için bekletilecek nav metnini güncelle.
     * Her yeni manevra talimatı söylendiğinde çağrılır.
     */
    fun setPendingNavigation(text: String?) {
        pendingNavText = text
    }

    // ─── Navigasyon Talimatı ─────────────────────────────────────────────────

    /**
     * Rota manevra talimatı.
     * QUEUE_FLUSH: Mevcut navBack kuyruğunu temizle, hemen söyle.
     * (YOLO uyarısı çalışırken bu çağrılamaz — MainActivity zaten engeller)
     */
    fun speakNavigation(text: String) {
        if (!isReady) {
            Log.w(TAG, "speakNavigation: TTS hazır değil, atlandı")
            return
        }
        if (text.isBlank()) return

        // Aynı metin zaten söyleniyorsa tekrar ekleme
        if (isSpeakingNavigation && currentNavText == text) {
            Log.d(TAG, "speakNavigation: duplikat engellendi")
            return
        }

        currentNavText = text
        val id = "NAV_${UUID.randomUUID()}"
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
        Log.d(TAG, "NAVİGASYON: ${text.take(80)}")
    }

    // ─── Temizlik ─────────────────────────────────────────────────────────────

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isReady = false
        pendingNavText = null
        currentNavText = null
    }
}
