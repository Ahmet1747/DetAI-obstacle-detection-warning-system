# DetAI - Görme Engelliler İçin Akıllı Asistan ve Navigasyon

DetAI, görme engelli bireylerin günlük hayatta bağımsız hareket edebilmelerini sağlamak amacıyla geliştirilmiş, yapay zeka destekli bir mobil uygulamadır. Görüntü işleme (YOLO) ve Google Haritalar entegrasyonunu bir araya getirerek, kullanıcılara hem çevresel engeller hakkında gerçek zamanlı bilgi verir hem de sesli komutlarla çalışan akıllı bir navigasyon deneyimi sunar.

## 🌟 Temel Özellikler

### 1. Akıllı Nesne ve Engel Tespiti
*   **Gerçek Zamanlı Tespit:** TensorFlow Lite ve YOLOv11/YOLOv8 modelleri kullanılarak kamera üzerinden anlık nesne tespiti yapılır.
*   **Risk Bazlı Önceliklendirme:** Ekranda tespit edilen tüm nesneler kullanıcıya okunmaz. Nesnenin tehlike seviyesine (örn: merdiven, araba = Yüksek; kedi, bank = Düşük) ve mesafesine göre bir **Risk Skoru** hesaplanır. En tehlikeli ve en yakın nesne anında seslendirilir.
*   **Dinamik Mesafe Hesaplama:** Cihazın kamera sensör özellikleri (focal length) dinamik olarak alınarak, pinhole kamera modeli ile nesnelerin uzaklığı metre ve santimetre cinsinden hassas bir şekilde ölçülür.
*   **Koltuk Doluluk Analizi:** Toplu taşıma veya park gibi alanlarda boş yer bulmayı kolaylaştırmak için, tespit edilen "Sandalye/Bank" ile "Kişi" (Person) nesnelerinin kesişimleri hesaplanır. Sistem kullanıcıya "3 metre mesafede boş sandalye" veya "dolu sandalye" şeklinde bilgi verir.
*   **Alan Taraması:** Kullanıcı ekrana çift dokunduğunda çevresindeki en yakın 3 nesneyi detaylı olarak tarar ve okur.

### 2. Sesli Komut ve Akıllı Navigasyon
*   **Doğal Dil İşleme ile Rota Çizme:** Ekrana uzun basılarak sesli komut asistanı aktif edilir. "Taksim'e rota çiz", "Kadıköy'e gidiyorum" gibi doğal Türkçe komutlar anlaşılarak hedef noktası belirlenir.
*   **Google API Entegrasyonu:** Geocoding ve Directions API'leri kullanılarak kullanıcının bulunduğu noktadan hedefe yaya rotası oluşturulur.
*   **Göreceli Yönlendirme:** Kullanıcının cihazdan aldığı pusula yönü (bearing) ile hedefin yönü karşılaştırılır. "Kuzeye gidin" yerine, kullanıcının o anki bakış açısına göre "Hafif sağa dönüp ilerleyin" şeklinde yönlendirmeler yapılır.
*   **Manevra Uyarıları:** Dönüş veya yön değiştirme noktalarına 25 metre kala otomatik olarak sesli bildirim yapılır.

### 3. Öncelikli Ses (TTS) Yönetimi (PriorityManager)
Uygulamanın en kritik bileşenlerinden biri ses kanalı yönetimidir:
*   Kullanıcı navigasyon talimatı dinlerken önüne ani bir engel çıkarsa (örn: Merdiven), navigasyon sesi **anında kesilir** ve engel uyarısı yapılır.
*   Tehlike uyarısı bittiğinde, yarım kalan navigasyon talimatı otomatik olarak tekrar okunur.

### 4. Güvenlik ve Uyarılar
*   **Pil Durumu Takibi:** Şarj seviyesi %15'e düştüğünde sistem kullanıcıyı uygulamanın kapanabileceği konusunda sesli olarak uyarır.
*   **GPS Kontrolü:** Konum servisi kapalıysa veya sinyal alınamıyorsa kullanıcı uyarılır ve gerekirse ayarlar menüsüne yönlendirilir.

## 🛠️ Kullanılan Teknolojiler ve Mimari

*   **Dil:** Kotlin
*   **Yapay Zeka & Görüntü İşleme:** TensorFlow Lite, YOLO (You Only Look Once), CameraX API
*   **Navigasyon:** Google Directions API, Google Geocoding API, FusedLocationProviderClient
*   **Ses İşleme:** Android TextToSpeech (TTS), SpeechRecognizer
*   **Ağ İşlemleri:** Retrofit2, OkHttp3, Gson, Kotlin Coroutines

## 📂 Proje Yapısı

*   `MainActivity.kt`: Kamera akışını, sesli komut dinlemeyi, GPS güncellemelerini ve nesne tespit sonuçlarının risk analizini (Smart Detection) yönetir.
*   `Detector.kt`: TensorFlow Lite modelini çalıştırır, NMS (Non-Maximum Suppression) uygular, dinamik mesafe hesabını ve koltuk doluluk oranını (IoU/Overlap) hesaplar.
*   `NavigationService.kt`: Retrofit ile Google API isteklerini yönetir, rotaları adımlara böler (ManeuverPoint) ve kullanıcının konumuna göre göreceli yön (sağ, sol, düz) hesaplamalarını yapar.
*   `PriorityManager.kt`: Çakışan sesli bildirimleri (YOLO engelleri vs. Navigasyon) yöneten özel bir TTS kuyruk sistemidir.
*   `OverlayView.kt`: Tespit edilen nesnelerin sınır kutularını (Bounding Box) ekranda görselleştirmek içindir.

## 🚀 Kurulum

1. Projeyi klonlayın.
2. `local.properties` dosyasına Google Haritalar API anahtarınızı ekleyin:
   ```properties
   GOOGLE_API_KEY=YOUR_API_KEY_HERE
   ```
3. Modellerin (örn: `yolov11n_float16.tflite`) `app/src/main/assets/` dizininde olduğundan emin olun.
4. Android Studio üzerinden `assembleDebug` ile build alıp cihazınızda çalıştırın.

---
*Görmek, sadece gözlerle yapılan bir eylem değildir.*
