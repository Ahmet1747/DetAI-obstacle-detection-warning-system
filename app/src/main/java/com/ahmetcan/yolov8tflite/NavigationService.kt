package com.ahmetcan.yolov8tflite

import android.util.Log
import android.text.Html
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.HttpException
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Query
import kotlin.math.*

// ─── Google Geocoding API ─────────────────────────────────────────────────────

interface GoogleGeocodingApi {
    @GET("maps/api/geocode/json")
    suspend fun geocode(
        @Query("address") address: String,
        @Query("language") language: String = "tr",
        @Query("key")     apiKey: String   = Constants.GOOGLE_API_KEY
    ): GeocodingResponse

    @GET("maps/api/geocode/json")
    suspend fun reverseGeocode(
        @Query("latlng") latlng: String,
        @Query("language") language: String = "tr",
        @Query("key")     apiKey: String   = Constants.GOOGLE_API_KEY
    ): GeocodingResponse
}

data class GeocodingResponse(
    @SerializedName("status")  val status: String?,
    @SerializedName("results") val results: List<GeocodingResult>?
)

data class GeocodingResult(
    @SerializedName("formatted_address") val formattedAddress: String?,
    @SerializedName("geometry")          val geometry: GeoGeometry?
)

data class GeoGeometry(
    @SerializedName("location") val location: LatLng?
)

data class LatLng(
    @SerializedName("lat") val lat: Double?,
    @SerializedName("lng") val lng: Double?
)

// ─── Google Directions API ────────────────────────────────────────────────────

interface GoogleDirectionsApi {
    @GET("maps/api/directions/json")
    suspend fun getDirections(
        @Query("origin")      origin: String,       // "lat,lng"
        @Query("destination") destination: String,  // "lat,lng"
        @Query("mode")        mode: String    = "walking",
        @Query("language")    language: String = "tr",
        @Query("key")         apiKey: String   = Constants.GOOGLE_API_KEY
    ): DirectionsResponse
}

data class DirectionsResponse(
    @SerializedName("status") val status: String?,
    @SerializedName("routes") val routes: List<DirectionsRoute>?
)

data class DirectionsRoute(
    @SerializedName("legs") val legs: List<DirectionsLeg>?
)

data class DirectionsLeg(
    @SerializedName("distance")      val distance: DirectionsValue?,
    @SerializedName("duration")      val duration: DirectionsValue?,
    @SerializedName("start_address") val startAddress: String?,
    @SerializedName("end_address")   val endAddress: String?,
    @SerializedName("steps")         val steps: List<DirectionsStep>?
)

data class DirectionsValue(
    @SerializedName("text")  val text: String?,
    @SerializedName("value") val value: Int?  // metre veya saniye
)

data class DirectionsStep(
    @SerializedName("html_instructions") val htmlInstructions: String?,
    @SerializedName("distance")          val distance: DirectionsValue?,
    @SerializedName("duration")          val duration: DirectionsValue?,
    @SerializedName("maneuver")          val maneuver: String?,
    @SerializedName("start_location")    val startLocation: LatLng?,
    @SerializedName("end_location")      val endLocation: LatLng?
)

// ─── Dahili Manevra Noktası ───────────────────────────────────────────────────

data class ManeuverPoint(
    val lat: Double,
    val lon: Double,
    val rawText: String,       // Google'dan gelen orijinal metin (pusula yönlü olabilir)
    val sign: Int,             // Google maneuver string → sayısal (bilgi amaçlı)
    val distanceToNext: Double,
    val targetBearing: Double, // Bu adımın hedef yönü (derece, 0=Kuzey)
    val hasCardinalDir: Boolean = false, // Pusula yönü içeriyor mu?
    var announced: Boolean = false
)

data class ManeuverResult(
    val text: String,
    val distanceMeters: Double
)

// ─── Navigasyon Servisi ───────────────────────────────────────────────────────

object NavigationService {

    private const val TAG = "NavigationService"

    private val httpClient by lazy {
        OkHttpClient.Builder()
            .addInterceptor(HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BASIC
            })
            .build()
    }

    private val geocodingApi: GoogleGeocodingApi by lazy {
        Retrofit.Builder()
            .baseUrl(Constants.GOOGLE_GEOCODING_BASE_URL)
            .client(httpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(GoogleGeocodingApi::class.java)
    }

    private val directionsApi: GoogleDirectionsApi by lazy {
        Retrofit.Builder()
            .baseUrl(Constants.GOOGLE_DIRECTIONS_BASE_URL)
            .client(httpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(GoogleDirectionsApi::class.java)
    }

    @Volatile internal var maneuverPoints: List<ManeuverPoint> = emptyList()

    @Volatile var totalDistanceM: Double = 0.0
        private set

    @Volatile var totalDurationS: Int = 0
        private set

    @Volatile var isRouteActive: Boolean = false
        private set

    @Volatile var lastDestinationName: String = ""
        private set

    /** Rota yüklenince hemen seslendirilecek ilk talimat */
    @Volatile var firstInstructionText: String = ""
        private set

    /** Kullanıcının şu an baktığı yön (GPS bearing, 0=Kuzey, 90=Doğu) */
    @Volatile var currentUserBearing: Float = 0f

    // ─── Geocoding ────────────────────────────────────────────────────────────

    /**
     * Yer adını Google Geocoding API ile koordinata çevirir.
     * @return Pair(lat, lon) veya bulunamazsa null
     */
    suspend fun geocodePlace(placeName: String): Pair<Double, Double>? =
        withContext(Dispatchers.IO) {
            try {
                val response = geocodingApi.geocode(address = placeName)
                Log.d("DetAI_Debug", "geocode status=${response.status} results=${response.results?.size}")

                if (response.status != "OK") {
                    Log.e(TAG, "Geocoding başarısız: status=${response.status}")
                    return@withContext null
                }

                val result = response.results?.firstOrNull() ?: return@withContext null
                val lat = result.geometry?.location?.lat ?: return@withContext null
                val lon = result.geometry?.location?.lng ?: return@withContext null

                lastDestinationName = result.formattedAddress
                    ?.split(",")?.firstOrNull()?.trim() ?: placeName

                Log.d(TAG, "Geocoding: \"$placeName\" → $lat,$lon ($lastDestinationName)")
                Log.d("DetAI_Debug", "geocodePlace → $lastDestinationName: lat=$lat lon=$lon")
                Pair(lat, lon)
            } catch (e: Exception) {
                Log.e(TAG, "Geocoding hatası: ${e.message}", e)
                null
            }
        }

    /**
     * Koordinatı adres metnine (sokak, mahalle vs.) çevirir.
     */
    suspend fun getCurrentAddressName(lat: Double, lon: Double): String? =
        withContext(Dispatchers.IO) {
            try {
                val response = geocodingApi.reverseGeocode(latlng = "$lat,$lon")
                Log.d("DetAI_Debug", "reverseGeocode status=${response.status}")

                if (response.status != "OK") return@withContext null

                val result = response.results?.firstOrNull() ?: return@withContext null
                val address = result.formattedAddress
                
                // Çok uzun olmasın diye adresin sadece ilk 2 virgüle kadar olan kısmını alıyoruz (Örn: "Caferağa, Moda Cd.")
                val shortAddress = address?.split(",")?.take(2)?.joinToString(",")?.trim()
                Log.d("DetAI_Debug", "Anlık konum: $shortAddress")
                shortAddress ?: address
            } catch (e: Exception) {
                Log.e(TAG, "Reverse Geocoding hatası: ${e.message}", e)
                null
            }
        }

    // ─── Rota İndirme ─────────────────────────────────────────────────────────

    /**
     * Google Directions API'den yürüyüş rotası indirir.
     * Adım talimatları ManeuverPoint listesine dönüştürülür.
     * @return Başarılı ise true
     */
    suspend fun fetchRoute(
        startLat: Double,
        startLon: Double,
        destLat: Double,
        destLon: Double
    ): Boolean = withContext(Dispatchers.IO) {
        val originStr = "$startLat,$startLon"
        val destStr   = "$destLat,$destLon"
        Log.d("DetAI_Debug", "fetchRoute → origin=$originStr dest=$destStr")

        try {
            val response = directionsApi.getDirections(
                origin      = originStr,
                destination = destStr
            )

            Log.d("DetAI_Debug", "fetchRoute status=${response.status}")

            if (response.status != "OK") {
                Log.e(TAG, "Directions API hatası: status=${response.status}")
                return@withContext false
            }

            val leg = response.routes?.firstOrNull()?.legs?.firstOrNull()
            if (leg == null) {
                Log.e(TAG, "fetchRoute: leg bulunamadı")
                return@withContext false
            }

            totalDistanceM = leg.distance?.value?.toDouble() ?: 0.0
            totalDurationS = leg.duration?.value ?: 0

            // Her adımı ManeuverPoint'e dönüştür
            val points = leg.steps?.mapNotNull { step ->
                val lat = step.startLocation?.lat ?: return@mapNotNull null
                val lon = step.startLocation?.lng ?: return@mapNotNull null
                val endLat = step.endLocation?.lat ?: lat
                val endLon = step.endLocation?.lng ?: lon
                val rawHtml = step.htmlInstructions ?: return@mapNotNull null
                // HTML tag'lerini temizle ("Sola dönün" → TTS'e gönderilir)
                val rawText = Html.fromHtml(rawHtml, Html.FROM_HTML_MODE_COMPACT)
                    .toString().trim()
                if (rawText.isBlank()) return@mapNotNull null

                // Bu adımın hedef yönünü hesapla (start → end)
                val bearing = bearingBetween(lat, lon, endLat, endLon)
                val hasCardinal = containsCardinalDirection(rawText)

                ManeuverPoint(
                    lat            = lat,
                    lon            = lon,
                    rawText        = rawText,
                    sign           = maneuverToSign(step.maneuver),
                    distanceToNext = step.distance?.value?.toDouble() ?: 0.0,
                    targetBearing  = bearing,
                    hasCardinalDir = hasCardinal
                )
            } ?: emptyList()

            maneuverPoints      = points
            isRouteActive       = points.isNotEmpty()

            // İlk adımın özetini oluştur
            firstInstructionText = buildRouteSummary(points)

            // Tüm adımları logla
            points.forEachIndexed { i, p ->
                Log.d("DetAI_Debug", "  Adım[$i] dist=${"%.0f".format(p.distanceToNext)}m bearing=${"%.0f".format(p.targetBearing)}°: ${p.rawText}")
            }
            Log.d("DetAI_Debug", "fetchRoute → BAŞARILI: ${points.size} adım, toplam ${"%.0f".format(totalDistanceM)}m")
            Log.d(TAG, "Rota indirildi. ${points.size} adım. Toplam: ${"%.0f".format(totalDistanceM)}m")
            true

        } catch (e: HttpException) {
            val body = e.response()?.errorBody()?.string() ?: "(boş)"
            Log.e(TAG, "fetchRoute HTTP ${e.code()}: $body")
            Log.d("DetAI_Debug", "fetchRoute HTTP HATA ${e.code()}: $body")
            false
        } catch (e: Exception) {
            Log.e(TAG, "fetchRoute hatası: ${e.message}", e)
            Log.d("DetAI_Debug", "fetchRoute HATA: ${e.javaClass.simpleName}: ${e.message}")
            false
        }
    }

    // ─── Manevra Yakınlık Kontrolü ────────────────────────────────────────────

    /**
     * GPS konumu güncellendikçe çağrılır.
     * Sıradaki seslendirilmemiş adıma yaklaşılınca döner.
     */
    fun checkManeuver(currentLat: Double, currentLon: Double): ManeuverResult? {
        if (!isRouteActive || maneuverPoints.isEmpty()) return null

        // 1. Hedefe doğrudan varış kontrolü (Kullanıcı kestirme yapmış veya adımları atlamış olabilir)
        val finalPoint = maneuverPoints.last()
        val distToFinal = haversineMeters(currentLat, currentLon, finalPoint.lat, finalPoint.lon)
        
        if (distToFinal <= Constants.MANEUVER_TRIGGER_DISTANCE_M) {
            isRouteActive = false
            maneuverPoints.forEach { it.announced = true } // Tüm adımları tamamlanmış say
            return ManeuverResult(text = "Hedefinize ulaştınız.", distanceMeters = 0.0)
        }

        // 2. Normal manevra adımı kontrolü
        for (point in maneuverPoints) {
            if (point.announced) continue

            val dist = haversineMeters(currentLat, currentLon, point.lat, point.lon)

            if (dist <= Constants.MANEUVER_TRIGGER_DISTANCE_M) {
                point.announced = true
                val spokenText = getRelativeText(point)
                Log.d(TAG, "Manevra tetiklendi (${dist.toInt()}m): $spokenText")
                return ManeuverResult(text = spokenText, distanceMeters = dist)
            } else {
                // Sıradaki adıma henüz ulaşılmadı → sonraki adımları kontrol etme
                break
            }
        }

        // 3. Normal akışla tüm adımlar tamamlandıysa
        val remaining = maneuverPoints.count { !it.announced }
        if (remaining == 0 && isRouteActive) {
            isRouteActive = false
            return ManeuverResult(text = "Hedefinize ulaştınız.", distanceMeters = 0.0)
        }

        return null
    }

    /** Rotayı sıfırla */
    fun clearRoute() {
        maneuverPoints       = emptyList()
        isRouteActive        = false
        totalDistanceM       = 0.0
        totalDurationS       = 0
        firstInstructionText = ""
        Log.d(TAG, "Rota temizlendi.")
    }

    /** İlk adımın özetini oluşturur */
    private fun buildRouteSummary(points: List<ManeuverPoint>): String {
        if (points.isEmpty()) return ""
        val p = points.first()
        val text = getRelativeText(p)
        val distText = if (p.distanceToNext > 0) " ${p.distanceToNext.toInt()} metre." else ""
        return "$text.$distText"
    }

    /**
     * Mevcut rota durumunu döndürür ("neredeyim" komutu için).
     * Kalan adım sayısı, sıradaki talimat ve tahmini kalan mesafe.
     */
    fun getRouteStatus(currentLat: Double, currentLon: Double): String? {
        if (!isRouteActive || maneuverPoints.isEmpty()) return null

        val nextPoint = maneuverPoints.firstOrNull { !it.announced }
        if (nextPoint == null) {
            return "Hedefinize ulaşmak üzeresiniz."
        }

        val distToNext = haversineMeters(currentLat, currentLon, nextPoint.lat, nextPoint.lon)
        val remaining = maneuverPoints.count { !it.announced }
        val nextText = getRelativeText(nextPoint)

        return "Rotanızda $remaining adım kaldı. " +
               "Sıradaki: $nextText. " +
               "Yaklaşık ${distToNext.toInt()} metre ilerde."
    }

    // ─── Yardımcılar ──────────────────────────────────────────────────────────

    /** Google maneuver string'ini sayısal işarete çevirir (bilgi amaçlı) */
    private fun maneuverToSign(maneuver: String?): Int = when (maneuver) {
        null, "straight"          ->  0
        "turn-right"              ->  2
        "turn-left"               -> -2
        "turn-sharp-right"        ->  3
        "turn-sharp-left"         -> -3
        "turn-slight-right"       ->  1
        "turn-slight-left"        -> -1
        "uturn-right","uturn-left"->  6
        "roundabout-right"        ->  5
        "roundabout-left"         -> -5
        "ramp-right"              ->  7
        "ramp-left"               -> -7
        "merge"                   ->  8
        "fork-right"              ->  4
        "fork-left"               -> -4
        else                      ->  0
    }

    /** Metin pusula yönü içeriyor mu kontrol et */
    private fun containsCardinalDirection(text: String): Boolean {
        val pattern = Regex("""(?i)(kuzey|güney|doğu|batı)""")
        return pattern.containsMatchIn(text)
    }

    /**
     * Pusula yönü içeren talimatları kullanıcının baktığı yöne göre
     * göreceli yöne (sağa/sola/düz) dönüştürür.
     *
     * Kullanıcı kuzeye bakıyorsa ve talimat "Doğuya gidin" diyorsa:
     *   targetBearing=90, userBearing=0 → fark=+90 → "Sağa dönüp ilerleyin"
     *
     * Pusula yönü içermeyen talimatlar ("Sola dönün" vb.) aynen korunur.
     */
    fun getRelativeText(point: ManeuverPoint): String {
        // Pusula yönü yoksa (zaten "sola dönün" gibi) aynen döndür
        if (!point.hasCardinalDir) return point.rawText

        val userBearing = currentUserBearing.toDouble()
        var diff = point.targetBearing - userBearing

        // -180..180 aralığına normalize et
        while (diff > 180) diff -= 360
        while (diff < -180) diff += 360

        val relativeDir = when {
            diff in -20.0..20.0    -> "Düz ilerleyin"
            diff in 20.0..60.0     -> "Hafif sağa dönüp ilerleyin"
            diff in 60.0..120.0    -> "Sağa dönüp ilerleyin"
            diff in 120.0..180.0   -> "Arkaya dönüp sağa ilerleyin"
            diff in -60.0..-20.0   -> "Hafif sola dönüp ilerleyin"
            diff in -120.0..-60.0  -> "Sola dönüp ilerleyin"
            else                   -> "Arkaya dönüp sola ilerleyin"
        }

        Log.d(TAG, "Yön hesabı: user=${"%.0f".format(userBearing)}° " +
                   "target=${"%.0f".format(point.targetBearing)}° " +
                   "diff=${"%.0f".format(diff)}° → $relativeDir")
        return relativeDir
    }

    /** İki GPS noktası arasındaki yönü (bearing) derece cinsinden hesaplar */
    private fun bearingBetween(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val dLon = Math.toRadians(lon2 - lon1)
        val lat1Rad = Math.toRadians(lat1)
        val lat2Rad = Math.toRadians(lat2)
        val y = sin(dLon) * cos(lat2Rad)
        val x = cos(lat1Rad) * sin(lat2Rad) - sin(lat1Rad) * cos(lat2Rad) * cos(dLon)
        return (Math.toDegrees(atan2(y, x)) + 360) % 360
    }

    /** İki GPS koordinatı arasındaki mesafeyi metre cinsinden hesaplar (Haversine) */
    fun haversineMeters(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val r    = 6_371_000.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a    = sin(dLat / 2).pow(2) +
                   cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) *
                   sin(dLon / 2).pow(2)
        return r * 2 * atan2(sqrt(a), sqrt(1 - a))
    }
}
