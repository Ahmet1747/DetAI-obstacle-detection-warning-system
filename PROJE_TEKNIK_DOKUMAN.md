# Görme Engelliler İçin Yapay Zeka Destekli Akıllı Asistan ve Navigasyon Sistemi
## Teknik ve Mimari Dokümantasyon

Bu doküman, projeyi sunacak veya savunacak kişinin projeye baştan sona teknik düzeyde hakim olması amacıyla hazırlanmıştır. Projedeki her bir modülün ne işe yaradığı ve hangi algoritmaların kullanıldığı detaylıca açıklanmıştır.

---

## 1. GENEL SİSTEM MİMARİSİ
Proje, görüntü işleme (AI) ve konum servislerinin eşzamanlı çalıştığı, modüler bir Android uygulamasıdır. Temel mimari 4 ana bacaktan oluşur:
*   **Girdi Katmanı (Sensörler):** `CameraX` ile çevreden alınan anlık görüntüler (30 FPS), `FusedLocationProvider` ile alınan GPS koordinatları ve `SpeechRecognizer` ile alınan sesli komutlar.
*   **Yapay Zeka Katmanı:** Kamera görüntülerini işleyip nesne çıkaran **YOLOv11 Nano (Float16)** TensorFlow Lite modeli.
*   **İş Mantığı Katmanı:** Bulunan nesnelerin mesafesini hesaplayan, tehlikesine göre puanlayan ve filtreleyen matematiksel algoritmalar (`Detector.kt` ve `MainActivity.kt`).
*   **Çıktı (Tepki) Katmanı:** Öncelik kurallarına göre `Text-To-Speech (TTS)` ile kullanıcıya sesli bildirim veren `PriorityManager`.

---

## 2. KULLANILAN ALGORİTMALAR VE GÖREVLERİ

### A. Görüntü İşleme ve Yapay Zeka Algoritmaları
**1. YOLO (You Only Look Once):**
*   **Ne Yapar?** Görüntüyü tek bir geçişte (grid'lere bölerek) analiz eder. Görüntüdeki nesnelerin etrafına sanal kutular (Bounding Box) çizer ve nesnenin ne olduğunu sınıflandırır. 
*   **Özelliği:** Projede YOLO'nun "Nano" versiyonu ve "Float16" (16-bit kuantizasyon) versiyonu kullanılmıştır. Bu sayede modelin boyutu ~5 MB'a düşürülmüş ve telefonda RAM/CPU tüketimi minimuma indirilerek gecikmesiz (<80ms) çalışması sağlanmıştır.

**2. Non-Maximum Suppression (NMS) Algoritması:**
*   **Ne Yapar?** Yapay zeka bazen aynı arabanın etrafına 5 tane farklı kutu çizebilir. NMS algoritması `Intersection over Union (IoU)` (Kesişim Bölü Birleşim) formülünü kullanarak üst üste binen kutuları tespit eder ve "En yüksek doğruluk oranına sahip tek bir kutuyu bırakıp diğerlerini siler."
*   **Amacı:** Kullanıcıya "Araba var, araba var, araba var" diye tekrarlayan hatalı bildirimler gitmesini önler.

**3. Pinhole Kamera Modeli (Dinamik Odak Uzaklığı):**
*   **Ne Yapar?** Projenin en güçlü matematiksel özelliklerinden biridir. Nesnenin kameraya olan uzaklığını hesaplar.
*   **Nasıl Çalışır?** Sabit bir değer kullanmak yerine Android `CameraCharacteristics` API'si ile telefonun kamerasının gerçek "Fiziksel Sensör Genişliğini (mm)" ve "Gerçek Odak Uzaklığını (mm)" çeker. Görüntüdeki nesnenin piksel boyutunu gerçek dünyadaki ortalama boyutu (örneğin insan 170cm, kapı 200cm) ile oranlayarak optik bir derinlik hesabı yapar. Cihaz değişse bile mesafe ölçümü doğru çalışır.

**4. Şekil Mantıklılık Filtresi (Plausibility Check):**
*   **Ne Yapar?** Yapay zekanın "False Positive" (Yanlış Tespit) yapmasını engeller. 
*   **Nasıl Çalışır?** Model örneğin kaldırımdaki düz bir çizgiyi "Kapı" sanabilir. Algoritma kutunun En-Boy oranına (Aspect Ratio) bakar. Eğer sistemin kapı sandığı kutu yatay bir dikdörtgense (genişlik > yükseklik), "Kapılar dikey olur, bu kapı olamaz" diyerek tespiti çöpe atar.

**5. Koltuk Doluluk Analizi (Overlap / Kesişim Algoritması):**
*   **Ne Yapar?** Çevredeki bankların veya sandalyelerin boş mu yoksa dolu mu olduğunu anlar.
*   **Nasıl Çalışır?** Ekranda tespit edilen "İnsan" kutusu ile "Koltuk" kutusunun alanlarını karşılaştırır. İnsan kutusu, koltuk kutusunun toplam yüzey alanının **en az %15'ini** kapsıyorsa o koltuğu otomatik olarak "Dolu" ilan eder.

### B. Akıllı Bildirim ve Önceliklendirme Algoritmaları (`MainActivity.kt`)
Sistem makine tüfeği gibi her gördüğünü söylemez, insan beyni gibi "önce hangisini söylemeliyim?" diye düşünür.

**1. Tehlike Skoru (Risk Puanlaması):**
*   Çevrede aynı anda insan, kedi ve araba varsa sistem hangisini söyleyecek? Her sınıfa bir hayati risk puanı atadık (Araba, direk, merdiven = 10 Puan | İnsan, kapı = 5 Puan | Kedi, bank = 2 Puan).
*   **Formül:** `Tehlike Skoru = (Nesne Puanı) / (Nesnenin Mesafesi)`
*   Bu formül sayesinde 5 metre uzaktaki bir araba, 1 metre yanınızdaki kediden her zaman daha yüksek riskli çıkar.

**2. Risk Sıralı Döngü (Filtreleme Ağı):**
*   Bulunan nesneler risklerine göre büyükten küçüğe dizilir. En riskliden başlanarak filtrelere sokulur:
    *   **Mesafe Filtresi:** Eğer sistem bu nesneyi daha önce söylediyse, aradaki mesafe en az **30 cm** değişmeden (yaklaşmadan/uzaklaşmadan) tekrar söylemez (Spam'i engeller).
    *   **Zaman Filtresi:** Eğer sistem bir tehlikeyi haber verdiyse, sonraki **2 saniye** boyunca başka hiçbir şeyi söylemez (Kulaklıkta ses karmaşasını önler).
*   Eğer en riskli nesne 30 cm hareket etmediği için filtreye takılırsa, algoritma pes etmez; döngüyü bozmadan listedeki 2. en riskli nesneye bakar. Onu geçerse onu okur.

**3. TTS Susturucu Koruması (Dinamik Bekleme):**
*   Sesli anons yapıldığı sırada kameradan daha riskli bir nesne gelirse mevcut cümlenin yarım kesilmemesi için, sistemin anons boyunca (TTS `isSpeaking` olduğu sürece) anlık gelen görüntüleri bekleterek sabırla okumanın bitmesini beklediği özelliktir.

### C. Navigasyon ve Sesli Komut Algoritmaları
**1. Doğal Dil İşleme (Regex ile Ek Silme):**
*   Kullanıcı "Kadıköy'e rota çiz", "Taksim'e git" dediğinde sistem sadece "Kadıköy" kelimesini almak zorundadır. Yazılan özel Regex (Düzenli İfade) algoritması Türkçe dil bilgisindeki ses uyumlarını ve kesme işaretlerini (e, a, ye, ya, de, da vb.) kelimenin sonundan tıraşlayarak pürüzsüz adres verisini çıkartır.

**2. Google Geocoding & Directions API:**
*   Bulunan yer adı `Retrofit` ile Google'a gönderilir, enlem/boylam (GPS) kordinatına çevrilir (Geocoding). Ardından kullanıcının bulunduğu yerden o hedefe "Yaya Navigasyonu (Walking)" rotası çizilir.

**3. Haversine Formülü (Çevrimdışı Mesafe Hesabı):**
*   Kullanıcı yürürken telefon Google'a saniyede bir istek atarsa Google API kotası biter. Çözüm olarak **Haversine formülü** kullanılır.
*   Bu trigonometrik formül, dünyanın yuvarlaklığını (yarıçapını) hesaba katarak kullanıcının mevcut GPS koordinatı ile döneceği sokağın (manevra noktasının) koordinatı arasındaki mesafeyi yerel (çevrimdışı) olarak hesaplar. Kullanıcı dönüşe 25 metre yaklaştığında sesli talimat okutulur.

### D. Veri Seti Optimizasyon Algoritması (Smart Balance)
**1. Zeki Dengeleme (Smart Balance Dataset):**
*   Yapay zekayı eğitmek için kullanılan veri setlerinde "İnsan" sayısı çokken "Direk" veya "Merdiven" sayısı çok azdır. Modeli direkt böyle eğitirsek model her şeye İnsan demeye başlar.
*   Projedeki Python scripti, rastgele resim silmek yerine şuna bakar: "Bu resmin içinde sadece insan ve araba mı var?" Cevap evetse o resimlerin %70'ini siler. "Bu resimde insan var ama köşede 1 tane de DİREK var." O zaman o resmi **asla silmez**.
*   Böylece nadir görülen engellerin verileri korunurken, baskın veriler azaltılmış ve model çok daha dengeli ve zeki şekilde (yolov11n_float16_new) eğitilmiştir.

---

## 3. PROJENİN ÖNE ÇIKAN ÖZGÜN YANLARI (SUNUM İÇİN VURGULAR)
1.  **Donanıma Duyarlı Mesafe:** Her telefonun kamera açısı farklıdır. Pinhole Kamera hesaplaması sayesinde uygulama her marka telefona kendini adapte eder.
2.  **Sadece Görmez, Yorumlar:** Uygulama sıradan nesne tespiti yapmaz, koltuğun boş mu dolu mu olduğunu uzaysal kesişim alanından bularak sosyal bir asistanlık yapar.
3.  **Aptal Bir Robot Değildir:** 5 metre ilerideki arabayla 1 metre ilerideki kediyi ayırır. Spama düşmemek için 30 cm ve 2 saniye kurallarıyla çok ciddi bir filtreleme sistemi kullanır.
4.  **Güvenlik:** Rota çizerken ağaçlara veya duvarlara değil, yaya geçidi ve kaldırımları baz alır. Telefon şarjı %15'e düştüğünde görme engelli bireye sesli kriz uyarısı yapar.

Bu mimari sayesinde proje, "sıradan bir öğrenci yapay zeka uygulamasından" çok, piyasaya çıkmaya hazır, engelli dostu ve donanımı efektif kullanan profesyonel bir yazılıma dönüşmüştür.
