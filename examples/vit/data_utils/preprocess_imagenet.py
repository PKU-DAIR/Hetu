import os
import shutil

src_dir = "/home/gehao/njw1123/merge/examples/vit/imagenet"
dst_dir = "/home/gehao/njw1123/imagenet/train"

os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if not fname.endswith(".JPEG"):
        continue

    parts = fname.split('_')
    if len(parts) < 2:
        continue

    # 提取 synset id，如 n01751748
    label = parts[-1].split('.')[0]

    class_dir = os.path.join(dst_dir, label)
    os.makedirs(class_dir, exist_ok=True)

    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(class_dir, fname)

    shutil.copy2(src_path, dst_path)  