package com.ahmetcan.yolov8tflite

import org.tensorflow.lite.DataType

object Constants {
    const val MODEL_PATH = "yolov11n_float16_new.tflite"
    val LABELS_PATH: String? = "example_label_file_new.txt"

    const val CONFIDENCE_THRESHOLD = 0.55f
    const val IOU_THRESHOLD = 0.5f

    const val INPUT_MEAN = 0f
    const val INPUT_STANDARD_DEVIATION = 255f

    val INPUT_IMAGE_TYPE: DataType = DataType.FLOAT32
    val OUTPUT_IMAGE_TYPE: DataType = DataType.FLOAT32

    // ─── Google API ───────────────────────────────────────────────────────────
    // Key local.properties → BuildConfig pipeline üzerinden gelir (git'e gitmez)
    val GOOGLE_API_KEY: String get() = BuildConfig.GOOGLE_API_KEY
    const val GOOGLE_DIRECTIONS_BASE_URL = "https://maps.googleapis.com/"
    const val GOOGLE_GEOCODING_BASE_URL  = "https://maps.googleapis.com/"

    // Navigasyon: Manevra noktasına bu mesafede (metre) yaklaşılınca seslen
    // Görme engelli bir yaya için 25m (~20 saniye yürüme mesafesi) idealdir
    const val MANEUVER_TRIGGER_DISTANCE_M = 25.0

    // Nesne tespiti: Aynı nesne tekrar seslenmesi için minimum mesafe değişimi (santimetre)
    const val OBJECT_DISTANCE_THRESHOLD = 30f
}
