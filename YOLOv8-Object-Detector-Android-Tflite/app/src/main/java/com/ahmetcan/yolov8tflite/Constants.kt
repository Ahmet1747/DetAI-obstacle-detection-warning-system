package com.ahmetcan.yolov8tflite

import org.tensorflow.lite.DataType

object Constants {
    const val MODEL_PATH = "yolov8n_int8.tflite"
    val LABELS_PATH: String? = null

    // --- EKSİK OLAN VE HATALARA YOL AÇAN DEĞERLER ---
    const val CONFIDENCE_THRESHOLD = 0.5f // Güven eşiği
    const val IOU_THRESHOLD = 0.5f       // Çakışma eşiği (NMS için)

    const val INPUT_MEAN = 0f
    const val INPUT_STANDARD_DEVIATION = 255f

    val INPUT_IMAGE_TYPE: DataType = DataType.FLOAT32
    val OUTPUT_IMAGE_TYPE: DataType = DataType.FLOAT32
}