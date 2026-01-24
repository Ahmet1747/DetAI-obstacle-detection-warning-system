package com.ahmetcan.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String?,
    private val detectorListener: DetectorListener,
    private val message: (String) -> Unit
) {

    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    // Filtrelenen 32 Nesne Listesi
    private val allowedClasses = listOf(
        "person", "bicycle", "motorcycle", "car", "bus", "truck", "dog", "cat",
        "chair", "bench", "couch", "potted plant", "bed", "dining table", "bottle",
        "traffic light", "fire hydrant", "stop sign", "parking meter",
        "tv", "laptop", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "umbrella", "bird", "mouse"
    )

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(Constants.INPUT_MEAN, Constants.INPUT_STANDARD_DEVIATION))
        .add(CastOp(Constants.INPUT_IMAGE_TYPE))
        .build()

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    init {
        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply{
            if(compatList.isDelegateSupportedOnThisDevice){
                val delegateOptions = compatList.bestOptionsForThisDevice
                this.addDelegate(GpuDelegate(delegateOptions))
            } else {
                this.setNumThreads(4)
            }
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape = interpreter.getOutputTensor(0)?.shape()

        labels.addAll(MetaData.extractNamesFromMetadata(model))
        if (labels.isEmpty()) {
            if (labelPath == null) {
                labels.addAll(MetaData.TEMP_CLASSES)
            } else {
                labels.addAll(MetaData.extractNamesFromLabelFile(context, labelPath))
            }
        }

        inputShape?.let {
            tensorWidth = it[1]
            tensorHeight = it[2]
            if (it[1] == 3) {
                tensorWidth = it[2]
                tensorHeight = it[3]
            }
        }

        outputShape?.let {
            numChannel = it[1]
            numElements = it[2]
        }
    }

    fun detect(frame: Bitmap) {
        if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) return

        var inferenceTime = SystemClock.uptimeMillis()
        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)
        val tensorImage = TensorImage(Constants.INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements),
            Constants.OUTPUT_IMAGE_TYPE
        )
        interpreter.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }
        detectorListener.onDetect(bestBoxes, inferenceTime)
    }

    private fun bestBox(array: FloatArray) : List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = Constants.CONFIDENCE_THRESHOLD
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel){
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxIdx != -1) {
                val clsName = labels[maxIdx]

                // --- Sınıf Filtreleme ---
                if (!allowedClasses.contains(clsName.lowercase())) continue

                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]

                // --- Konum Belirleme ---
                val position = when {
                    cx < 0.35f -> "Solunda"
                    cx > 0.65f -> "Sağında"
                    else -> "Önünde"
                }

                // --- Mesafe Hesaplama ---
                val realWidth = getRealWidth(clsName)
                // 500f odak uzaklığıdır (Focal Length). Sapma varsa bu sayıyı değiştirebilirsin.
                val dist = (realWidth * 500f) / (w * tensorWidth)

                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)

                if (x1 < 0F || x1 > 1F || y1 < 0F || y1 > 1F) continue

                boundingBoxes.add(
                    BoundingBox(x1, y1, x2, y2, cx, cy, w, h, maxConf, maxIdx, clsName, dist, position)
                )
            }
        }
        return if (boundingBoxes.isEmpty()) null else applyNMS(boundingBoxes)
    }

    // 32 Nesnenin tamamı için Genişlik Tablosu
    private fun getRealWidth(label: String): Float {
        return when (label.lowercase()) {
            "person" -> 50f
            "car" -> 180f
            "bus", "truck" -> 250f
            "bicycle", "motorcycle" -> 60f
            "dog", "cat" -> 25f
            "bird" -> 15f
            "traffic light" -> 30f
            "stop sign" -> 75f
            "fire hydrant" -> 40f
            "parking meter" -> 35f
            "tv", "laptop" -> 40f
            "cell phone", "mouse" -> 7f
            "microwave", "oven" -> 55f
            "toaster" -> 25f
            "refrigerator" -> 70f
            "sink" -> 50f
            "bottle", "cup" -> 8f
            "chair" -> 45f
            "bench" -> 120f
            "couch", "bed", "dining table" -> 150f
            "potted plant" -> 30f
            "toilet" -> 45f
            "book", "clock", "vase" -> 15f
            "umbrella" -> 100f
            else -> 20f
        }
    }

    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()
        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)
            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                if (calculateIoU(first, nextBox) >= Constants.IOU_THRESHOLD) iterator.remove()
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    fun close() = interpreter.close()
    fun restart(isGpu: Boolean) { interpreter.close() }
}