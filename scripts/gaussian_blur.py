import cv2
import os
import albumentations as A
from tqdm import tqdm
import shutil

transform = A.Compose([
    A.GaussianBlur(blur_limit=(0, 5), p=0.5),
])

dataset_name = "dataset_split"
output_dataset_name = "blur_dataset_split"
dirs = ["/train/images", "/val/images", "/test/images"]
labels_dirs = ["/train/labels", "/val/labels", "/test/labels"]

# copy data.yaml
os.makedirs(output_dataset_name, exist_ok=True)

yaml_name = os.path.join(dataset_name, "data.yaml")
if (os.path.isfile(yaml_name)):
            if yaml_name.endswith('.yaml'):
                shutil.copy(yaml_name, output_dataset_name + "/")

for i, dir in enumerate(dirs):
    os.makedirs(output_dataset_name + dir, exist_ok=True)

    # copy labels
    inputlabeldir = dataset_name + labels_dirs[i]
    labeldir = output_dataset_name + labels_dirs[i]
    os.makedirs(labeldir, exist_ok=True)
    labels_filenames = [os.path.join(inputlabeldir, f) for f in os.listdir(inputlabeldir) if os.path.isfile(os.path.join(inputlabeldir, f))]
    for label in labels_filenames:
         shutil.copy(label, labeldir)


    for img_name in tqdm(os.listdir(dataset_name + dir)):

        img_path = os.path.join(dataset_name + dir, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = transform(image=img_rgb)['image']
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        # only augment the train set
        if i==0:
            cv2.imwrite(os.path.join(output_dataset_name + dir, img_name), augmented)
        else:
            cv2.imwrite(os.path.join(output_dataset_name + dir, img_name), img)