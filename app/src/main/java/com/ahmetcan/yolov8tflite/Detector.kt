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
    private var _tensorWidth = 0
    val tensorWidth: Int get() = _tensorWidth
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    private var isChannelLast = false

    // Odak uzaklığı (piksel cinsinden). Varsayılan 500f.
    // MainActivity tarafından CameraCharacteristics'ten hesaplanan gerçek değer atanır.
    @Volatile
    var focalLengthPx: Float = 500f


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
        
        // GPU desteklenmiyor veya model GPU'da hata veriyorsa CPU'ya fallback yap
        interpreter = try {
            Interpreter(model, options)
        } catch (e: Exception) {
            val fallbackOptions = Interpreter.Options().apply { setNumThreads(4) }
            Interpreter(model, fallbackOptions)
        }

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
            _tensorWidth = it[1]
            tensorHeight = it[2]
            if (it[1] == 3) {
                _tensorWidth = it[2]
                tensorHeight = it[3]
            }
        }

        outputShape?.let {
            if (it.size >= 3) {
                if (it[1] > it[2]) {
                    // format is [1, 8400, 84] -> channel last
                    numElements = it[1]
                    numChannel = it[2]
                    isChannelLast = true
                } else {
                    // format is [1, 84, 8400] -> channel first
                    numChannel = it[1]
                    numElements = it[2]
                    isChannelLast = false
                }
            } else {
                numChannel = it[1]
                numElements = it[2]
            }
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

        // Dinamik DataType: model int8/float16 ne veriyorsa onu kullan
        val outputDataType = interpreter.getOutputTensor(0)?.dataType() ?: Constants.OUTPUT_IMAGE_TYPE

        // Array is always 1D when flattened, create buffer with correct size regardless of dimension ordering
        val output = TensorBuffer.createFixedSize(intArrayOf(1, if(isChannelLast) numElements else numChannel, if(isChannelLast) numChannel else numElements),
            outputDataType
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
            var maxConf = 0f // Önce mutlak en yüksek olasılığa sahip sınıfı buluyoruz
            var maxIdx = -1
            var j = 4
            
            while (j < numChannel){
                val arrayIdx = if (isChannelLast) c * numChannel + j else c + numElements * j
                if (arrayIdx < array.size && array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
            }

            if (maxIdx != -1 && maxIdx < labels.size) {
                val clsName = labels[maxIdx]
                val threshold = getClassThreshold(clsName)

                // 1. Sınıfa özel güvenirlik eşiği kontrolü
                if (maxConf >= threshold) {
                    val cxIdx = if (isChannelLast) c * numChannel + 0 else c + numElements * 0
                    val cyIdx = if (isChannelLast) c * numChannel + 1 else c + numElements * 1
                    val wIdx  = if (isChannelLast) c * numChannel + 2 else c + numElements * 2
                    val hIdx  = if (isChannelLast) c * numChannel + 3 else c + numElements * 3

                    val cx = array[cxIdx]
                    val cy = array[cyIdx]
                    val w = array[wIdx]
                    val h = array[hIdx]

                    // 2. Fiziksel Mantıklılık (Shape Plausibility) Kontrolü
                    // Model düz yolu kediye veya kapıya benzetebiliyor. Kutu şekillerine bakarak sahteleri eliyoruz.
                    if (!isPlausibleShape(clsName, w, h)) continue

                // --- Konum Belirleme ---
                val position = when {
                    cx < 0.35f -> "Solunda"
                    cx > 0.65f -> "Sağında"
                    else -> "Önünde"
                }

                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)

                // --- Mesafe Hesaplama (Genişlik + Yükseklik Tabanlı Pinhole Kamera Modeli) ---
                val realWidth = getRealWidth(clsName)
                val realHeight = getRealHeight(clsName)
                val pixelWidth = w * tensorWidth
                val pixelHeight = h * tensorHeight

                // Genişlik ve yükseklikten ayrı ayrı mesafe hesapla
                val distW = if (pixelWidth > 1f) (realWidth * focalLengthPx) / pixelWidth else 5000f
                val distH = if (pixelHeight > 1f) (realHeight * focalLengthPx) / pixelHeight else 5000f

                // İkisinden KÜÇÜĞÜNÜ al: büyük olan çerçeve dışına taşmış/hatalı olabilir
                // Küçük olan genelde daha doğru (nesne tam görünüyor demektir)
                val dist = minOf(distW, distH).coerceIn(5f, 5000f)

                // KUTUYU EKRAN SINIRLARINA KIRP (Clamping)
                // Önceden x1 < 0 vb. durumlarda nesne tamamen yoksayılıyordu. Bu yüzden ekranı
                // kaplayan büyük nesneler (kaldırım, merdiven, otobüs) tespit edilemiyordu.
                val clampedX1 = x1.coerceIn(0F, 1F)
                val clampedY1 = y1.coerceIn(0F, 1F)
                val clampedX2 = x2.coerceIn(0F, 1F)
                val clampedY2 = y2.coerceIn(0F, 1F)

                // Kutu tamamen ekran dışındaysa (mantıksız koordinatlar) atla
                if (clampedX1 >= clampedX2 || clampedY1 >= clampedY2) continue

                boundingBoxes.add(
                    BoundingBox(clampedX1, clampedY1, clampedX2, clampedY2, cx, cy, w, h, maxConf, maxIdx, clsName, dist, position)
                )
                }
            }
        }
        if (boundingBoxes.isEmpty()) return null

        val nmsResult = applyNMS(boundingBoxes)
        // NMS sonrası koltuk doluluk analizi
        return analyzeSeatOccupancy(nmsResult)
    }

    /**
     * Koltuk / Sandalye Doluluk Analizi
     *
     * chair ve bench kutularını person kutularıyla karşılaştırır.
     * Eğer bir koltuk kutusu, herhangi bir person kutusuyla %15+ kesişiyorsa → DOLU
     * Kesişmiyorsa → BOŞ
     *
     * Bu sayede hem günlük hayattaki sandalye/bank, hem de otobüs/metro
     * koltuklarının boş mu dolu mu olduğu otomatik tespit edilir.
     */
    private fun analyzeSeatOccupancy(boxes: MutableList<BoundingBox>): MutableList<BoundingBox> {
        val seatLabels = setOf("chair", "bench")
        val seats   = boxes.filter { it.clsName.lowercase() in seatLabels }
        val persons = boxes.filter { it.clsName.lowercase() == "person" }

        for (seat in seats) {
            var occupied = false
            for (person in persons) {
                // Önceden IoU (Kesişim / Toplam Alan) kullanılıyordu. İnsan kutusu çok büyük
                // olduğunda oran küçülüyor ve yanlışlıkla 'boş' diyordu.
                // Artık sadece koltuğun alanının ne kadarının kaplandığına bakıyoruz.
                val overlap = calculateSeatOverlap(seat, person)
                
                // Eğer kişi, koltuğun en az %15'ini kapatıyorsa veya üstündeyse dolu kabul et
                if (overlap >= 0.15f) {
                    occupied = true
                    break
                }
            }
            seat.isOccupied = occupied
        }
        return boxes
    }

    private fun calculateSeatOverlap(seat: BoundingBox, person: BoundingBox): Float {
        val x1 = maxOf(seat.x1, person.x1)
        val y1 = maxOf(seat.y1, person.y1)
        val x2 = minOf(seat.x2, person.x2)
        val y2 = minOf(seat.y2, person.y2)
        
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val seatArea = (seat.x2 - seat.x1) * (seat.y2 - seat.y1)
        
        if (seatArea <= 0F) return 0F
        return intersectionArea / seatArea
    }

    // Nesnelerin ortalama gerçek GENİŞLİKLERİ (cm) — kameraya bakan yüz genişliği
    private fun getRealWidth(label: String): Float {
        return when (label.lowercase()) {
            "bench"         -> 120f
            "bicycle"       -> 55f
            "bus"           -> 255f
            "car"           -> 180f
            "cat"           -> 25f
            "chair"         -> 50f
            "crosswalk"     -> 250f
            "dog"           -> 30f
            "door"          -> 90f
            "motorcycle"    -> 65f
            "person"        -> 45f
            "pole"          -> 12f
            "sidewalk"      -> 150f
            "stair"         -> 100f
            "stop sign"     -> 75f
            "traffic light" -> 30f
            "truck"         -> 250f
            else            -> 30f
        }
    }

    // Nesnelerin ortalama gerçek YÜKSEKLİKLERİ (cm) — mesafe doğruluğu ikinci kontrol
    private fun getRealHeight(label: String): Float {
        return when (label.lowercase()) {
            "bench"         -> 80f
            "bicycle"       -> 100f
            "bus"           -> 300f
            "car"           -> 150f
            "cat"           -> 25f
            "chair"         -> 90f
            "crosswalk"     -> 15f   // çizgiler alçak, yükseklik güvenilmez → distW kullanılır
            "dog"           -> 40f
            "door"          -> 200f
            "motorcycle"    -> 110f
            "person"        -> 170f
            "pole"          -> 300f
            "sidewalk"      -> 10f   // düz yüzey, yükseklik güvenilmez → distW kullanılır
            "stair"         -> 150f
            "stop sign"     -> 75f
            "traffic light" -> 80f
            "truck"         -> 350f
            else            -> 50f
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
        val unionArea = box1Area + box2Area - intersectionArea
        if (unionArea <= 0F) return 0F
        return intersectionArea / unionArea
    }

    private fun getClassThreshold(label: String): Float {
        return when (label.lowercase()) {
            "cat" -> 0.75f      // Kediler yola benzetildiği için çok yüksek eminlik iste
            "door" -> 0.75f     // Kapılar yola benzetildiği için yüksek eminlik iste
            "car" -> 0.40f      // Recall %34 → güvenlik açısından kritik, eşiği düşür
            "person" -> 0.40f   // Recall %37 → güvenlik açısından kritik, eşiği düşür
            "bicycle" -> 0.40f  // Recall düşük, eşiği düşür
            "chair" -> 0.40f    // Recall %36 → koltuk doluluk tespiti için önemli
            "motorcycle" -> 0.45f // Recall düşük
            "traffic light" -> 0.45f // Recall %48, biraz düşür
            "sidewalk" -> 0.35f // Kaldırımlar zor tespit ediliyor, eşiği daha da düşür
            "pole" -> 0.45f     // Direkler ince ve zor tespit ediliyor, eşiği düşür
            "crosswalk" -> 0.45f // Yaya geçitleri de zemin olduğu için zor
            "stair" -> 0.45f    // Merdivenler de benzer şekilde
            else -> Constants.CONFIDENCE_THRESHOLD // Varsayılan 0.55f
        }
    }

    private fun isPlausibleShape(label: String, w: Float, h: Float): Boolean {
        if (h <= 0f) return false
        val aspectRatio = w / h 
        
        return when (label.lowercase()) {
            "door" -> {
                // Kapılar dikeydir. Kare veya yatay dikdörtgen şeklindeki zemin fayanslarını elemek için
                // genişlik/yükseklik oranını 0.7 veya altına indiriyoruz.
                aspectRatio < 0.7f 
            }
            "cat" -> {
                // Kedi ekranın %70'ini kaplayacak kadar devasa yatay bir kutu olamaz (yol hatası)
                w < 0.7f && h < 0.7f
            }
            "pole" -> {
                // Direkler genelde incedir ama yakından veya eğik açıdan daha geniş görünebilir
                aspectRatio < 1.2f
            }
            else -> true
        }
    }

    fun close() = interpreter.close()
}