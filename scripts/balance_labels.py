"""
YOLO Veri Seti Dengeleme Scripti
================================
COCO'dan çekilen veri setlerinde person/car gibi baskın sınıfların
etiketlerini sınırlandırır.

Kullanım:
  python balance_labels.py --labels_dir ./train/labels --class_limits "10:6000,3:5000,5:4000"

Ne yapar:
  1. Tüm label dosyalarını tarar
  2. Her sınıfın toplam instance sayısını hesaplar
  3. Limiti aşan sınıfların fazla etiketlerini GÖRSELDEN SİLER
  4. Eğer bir görselde sadece silinen sınıflar varsa, görseli ve labeli tamamen kaldırır
"""

import os
import random
import argparse
import shutil
from collections import defaultdict
from pathlib import Path


def long_path(p):
    """Windows'ta 260 karakter limitini asmak icin \\\\?\\ prefix ekle"""
    s = str(p)
    if os.name == 'nt' and not s.startswith('\\\\?\\'):
        s = '\\\\?\\' + os.path.abspath(s)
    return s


def count_instances(labels_dir):
    """Her sinifin toplam instance sayisini hesapla"""
    counts = defaultdict(int)
    for label_file in Path(labels_dir).glob("*.txt"):
        with open(long_path(label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    counts[class_id] += 1
    return dict(counts)


def balance_labels(labels_dir, images_dir, class_limits, dry_run=False):
    """
    Fazla instance'ları olan sınıfları sınırla.
    
    class_limits: {class_id: max_instances} dict
    """
    
    # 1. Mevcut durumu say
    print("\n[MEVCUT DURUM]")
    counts = count_instances(labels_dir)
    for cls_id, count in sorted(counts.items()):
        limit = class_limits.get(cls_id)
        status = f" -> LIMIT: {limit}" if limit and count > limit else ""
        print(f"  Sinif {cls_id}: {count} instance{status}")
    
    # 2. Limiti aşan sınıflar için hangi dosyalarda kaç instance var?
    over_limit_classes = {
        cls_id: limit for cls_id, limit in class_limits.items()
        if counts.get(cls_id, 0) > limit
    }
    
    if not over_limit_classes:
        print("\n[OK] Hicbir sinif limitin uzerinde degil. Islem gerekmiyor.")
        return
    
    print(f"\n[SINIRLANDIRILACAK SINIFLAR]: {over_limit_classes}")
    
    # 3. Her dosyadaki her sınıfın instance'larını topla
    file_instances = {}  # {filepath: [(line_idx, class_id, line_text), ...]}
    
    for label_file in sorted(Path(labels_dir).glob("*.txt")):
        instances = []
        with open(long_path(label_file), 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    instances.append((idx, class_id, line.strip()))
        file_instances[label_file] = instances
    
    # 4. Limiti aşan her sınıf için rastgele instance'ları seç ve sil
    for cls_id, limit in over_limit_classes.items():
        current = counts[cls_id]
        to_remove = current - limit
        
        # Bu sınıfın tüm instance'larını (dosya, satır) olarak topla
        all_instances = []
        for filepath, instances in file_instances.items():
            for (line_idx, c_id, line_text) in instances:
                if c_id == cls_id:
                    all_instances.append((filepath, line_idx))
        
        # Rastgele silinecekleri seç
        random.shuffle(all_instances)
        to_delete = set()
        for item in all_instances[:to_remove]:
            to_delete.add(item)
        
        # file_instances'tan kaldır
        for filepath in file_instances:
            file_instances[filepath] = [
                (idx, c_id, text) for (idx, c_id, text) in file_instances[filepath]
                if (filepath, idx) not in to_delete
            ]
        
        print(f"  Sinif {cls_id}: {current} -> {limit} ({to_remove} instance silindi)")
    
    # 5. Dosyaları yeniden yaz
    removed_files = 0
    modified_files = 0
    
    for filepath, instances in file_instances.items():
        if not instances:
            # Boş kalan label dosyasını ve karşılık gelen görseli sil
            if not dry_run:
                os.remove(long_path(filepath))
                # Karşılık gelen görseli de sil
                if images_dir:
                    stem = filepath.stem
                    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                        img_path = Path(images_dir) / f"{stem}{ext}"
                        if os.path.exists(long_path(img_path)):
                            os.remove(long_path(img_path))
                            break
            removed_files += 1
        else:
            # Değişen dosyayı yeniden yaz
            new_content = "\n".join([text for (_, _, text) in instances]) + "\n"
            
            # Orijinalle karşılaştır
            with open(long_path(filepath), 'r') as f:
                old_content = f.read()
            
            if new_content != old_content:
                if not dry_run:
                    with open(long_path(filepath), 'w') as f:
                        f.write(new_content)
                modified_files += 1
    
    # 6. Sonuç
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Sonuç:")
    print(f"  Degistirilen label dosyasi: {modified_files}")
    print(f"  Silinen dosya cifti (label+gorsel): {removed_files}")
    
    # Son durumu göster
    if not dry_run:
        print("\n[YENI DURUM]")
        new_counts = count_instances(labels_dir)
        for cls_id, count in sorted(new_counts.items()):
            print(f"  Sinif {cls_id}: {count} instance")


def main():
    parser = argparse.ArgumentParser(description='YOLO veri seti dengeleme')
    parser.add_argument('--labels_dir', required=True, help='Label dosyalarının dizini')
    parser.add_argument('--images_dir', default=None, help='Görsel dosyalarının dizini (opsiyonel)')
    parser.add_argument('--class_limits', required=True,
                       help='Sınıf limitleri. Format: "class_id:limit,class_id:limit" Örn: "10:6000,3:5000"')
    parser.add_argument('--dry_run', action='store_true', help='Dosyaları değiştirmeden sadece göster')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (tekrarlanabilirlik için)')
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # class_limits parse
    class_limits = {}
    for pair in args.class_limits.split(','):
        cls_id, limit = pair.split(':')
        class_limits[int(cls_id)] = int(limit)
    
    print("=" * 50)
    print("YOLO Veri Seti Dengeleme")
    print("=" * 50)
    print(f"Labels: {args.labels_dir}")
    print(f"Images: {args.images_dir}")
    print(f"Limitler: {class_limits}")
    
    balance_labels(args.labels_dir, args.images_dir, class_limits, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
