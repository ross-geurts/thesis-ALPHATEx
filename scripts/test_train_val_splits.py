import os
import shutil
import random

source_images_dir = 'dataset_bin_fullwithglobal/train/images'
source_labels_dir = 'dataset/train/labels'

output_base_dir = 'dataset_fullwithglobal_split'

split_ratios = {'train':0.8, 'val':0.1, 'test': 0.1}
seed = 42

image_filenames = [f for f in os.listdir(source_images_dir) if os.path.isfile(os.path.join(source_images_dir, f))]
image_filenames.sort()
random.seed(seed)
random.shuffle(image_filenames)

total = len(image_filenames)
train_end = int(split_ratios['train'] * total)
val_end = train_end + int(split_ratios['val'] * total)

splits = {
    'train': image_filenames[:train_end],
    'val': image_filenames[train_end:val_end],
    'test': image_filenames[val_end:]
}

for split in splits:
    os.makedirs(os.path.join(output_base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, split, 'labels'), exist_ok=True)

for split, files in splits.items():
    for filename in files:
        image_src = os.path.join(source_images_dir, filename)
        label_src = os.path.join(source_labels_dir, os.path.splitext(filename)[0] + '.txt')

        image_dst = os.path.join(output_base_dir, split, 'images', filename)
        label_dst = os.path.join(output_base_dir, split, 'labels', os.path.splitext(filename)[0] + '.txt')

        shutil.copy2(image_src, image_dst)
        if os.path.exists(label_src):
            shutil.copy2(label_src, label_dst)
        else:
            print(f"Warning: Label file missing for {filename}")

print("Dataset split completed.")