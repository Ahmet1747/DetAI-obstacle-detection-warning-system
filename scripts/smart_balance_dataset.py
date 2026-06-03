import os
import random
import argparse
import shutil
from collections import defaultdict
from pathlib import Path

"""
AKILLI VERİ SETİ DENGELEME SCRİPTİ
==================================

Eski yaklaşım (balance_labels.py) neden modelinizi bozdu?
Eski kod, bir resimdeki örneğin 10 arabadan 5'inin etiketini (bounding box) siliyordu.
Ancak resimde o 5 araba hala duruyordu! YOLO modeli bu etiketsiz arabaları gördüğünde,
"Demek ki bunlar araba değil, arka plan" diye öğrendi. Bu yüzden Confusion Matrix'te 
insan, araba ve sandalyelerin %50'den fazlası "Background" (Arka plan) olarak algılandı.

YENİ YAKLAŞIM: GÖRSEL BAZLI EKSİLTME (Image-level undersampling)
1. Eğer bir görsel sadece çok fazla olan sınıfları (insan, araba) içeriyorsa, görseli komple sileriz.
2. Eğer görselde çok olan bir sınıfın yanında "nadir/önemli" bir sınıf da varsa (örn: insan ve yaya geçidi), o görsel KESİNLİKLE silinmez.
3. Hiçbir görselin içinden tekil etiket silinmez, görsel ya tamamen kalır ya da tamamen silinir. Etiketsiz nesne (hayalet nesne) oluşmaz.
"""

def long_path(p):
    s = str(p)
    if os.name == 'nt' and not s.startswith('\\\\?\\'):
        s = '\\\\?\\' + os.path.abspath(s)
    return s

def count_instances_and_images(labels_dir):
    """Sınıf sayılarını ve hangi görselde hangi sınıfların olduğunu bulur"""
    class_counts = defaultdict(int)
    image_classes = defaultdict(set)
    
    for label_file in Path(labels_dir).glob("*.txt"):
        with open(long_path(label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    image_classes[label_file].add(class_id)
                    
    return dict(class_counts), image_classes

def smart_balance(labels_dir, images_dir, dominant_classes, keep_probability=0.3, dry_run=False):
    """
    dominant_classes: Fazla olan ve azaltmak istediğimiz sınıf ID'leri (örn: [0, 2] -> insan ve araba)
    keep_probability: Sadece dominant sınıfları içeren resimlerin yüzde kaçını tutacağımız (0.3 = %30'unu tut, %70'ini sil)
    """
    
    print("\n[MEVCUT DURUM]")
    counts, image_classes = count_instances_and_images(labels_dir)
    for cls_id, count in sorted(counts.items()):
        print(f"  Sınıf {cls_id}: {count} etiket")
        
    dominant_set = set(dominant_classes)
    
    to_delete_files = []
    
    for filepath, classes_in_image in image_classes.items():
        # Görseldeki sınıfların HEPSİ dominant sınıflar içindeyse (örn: sadece insan veya sadece insan+araba varsa)
        if classes_in_image.issubset(dominant_set):
            # Rastgele zar at, eğer şans tutmazsa bu resmi tamamen sil
            if random.random() > keep_probability:
                to_delete_files.append(filepath)
                
    print(f"\n[ANALİZ SONUCU]")
    print(f"  Toplam görsel sayısı: {len(image_classes)}")
    print(f"  Sadece dominant sınıfları içeren ve SİLİNECEK görsel sayısı: {len(to_delete_files)}")
    
    if not dry_run:
        deleted_count = 0
        for filepath in to_delete_files:
            # Label dosyasını sil
            if os.path.exists(long_path(filepath)):
                os.remove(long_path(filepath))
            
            # Resim dosyasını sil (uzantıyı tahmin et)
            if images_dir:
                stem = filepath.stem
                for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                    img_path = Path(images_dir) / f"{stem}{ext}"
                    if os.path.exists(long_path(img_path)):
                        os.remove(long_path(img_path))
                        break
            
            deleted_count += 1
        
        print(f"\n[BAŞARILI] {deleted_count} görsel ve etiket dosyası silindi.")
        
        print("\n[YENİ DURUM]")
        new_counts, _ = count_instances_and_images(labels_dir)
        for cls_id, count in sorted(new_counts.items()):
            print(f"  Sınıf {cls_id}: {count} etiket")
    else:
        print("\n[DRY RUN] Herhangi bir dosya silinmedi. Gerçekten silmek için --dry_run parametresini kaldırın.")

def main():
    parser = argparse.ArgumentParser(description='YOLO Akıllı Veri Seti Dengeleme')
    parser.add_argument('--labels_dir', required=True, help='Label (.txt) dosyalarının klasörü')
    parser.add_argument('--images_dir', required=True, help='Resim dosyalarının klasörü')
    parser.add_argument('--dominant_classes', required=True, help='Azaltılacak sınıfların ID leri, virgülle ayırın (Örn: 0,2,56)')
    parser.add_argument('--keep_prob', type=float, default=0.3, help='Dominant resimlerin hayatta kalma ihtimali (0.0 ile 1.0 arası)')
    parser.add_argument('--dry_run', action='store_true', help='Dosyaları silmeden sadece ne olacağını göster')
    
    args = parser.parse_args()
    
    dom_classes = [int(x.strip()) for x in args.dominant_classes.split(',')]
    
    print("=" * 50)
    print("YOLO Akıllı Görsel Bazlı Dengeleme")
    print("=" * 50)
    
    smart_balance(args.labels_dir, args.images_dir, dom_classes, args.keep_prob, args.dry_run)

if __name__ == '__main__':
    main()
