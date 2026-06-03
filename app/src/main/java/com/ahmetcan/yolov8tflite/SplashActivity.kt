package com.ahmetcan.yolov8tflite

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity
import android.view.animation.AnimationUtils
import android.widget.ImageView
// Diğer importlar (Intent, Handler vb.)
class SplashActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        // 1. Logoyu XML'deki ID'si ile bul (activity_splash.xml'de android:id="@+id/logo_image" olmalı)
        val logo = findViewById<ImageView>(R.id.logo_image)

        // 2. Animasyon dosyasını yükle
        val splashAnim = AnimationUtils.loadAnimation(this, R.anim.splash_anim)

        // 3. Animasyonu logoya uygula
        logo.startAnimation(splashAnim)

        // 3.5 saniye bekle ve ana ekrana geç (Animasyonun bitmesi için süreyi biraz artırdık)
        Handler(Looper.getMainLooper()).postDelayed({
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)

            // Geçiş animasyonu (Opsiyonel: Ekranın sönerek değişmesini sağlar)
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)

            finish()
        }, 3500)
    }
}