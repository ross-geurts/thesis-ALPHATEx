import cv2
import os
import numpy as np
import shutil

# at first an itensity histogram  H={f1, f2 ..., f255} is computed, 
# where each fn represents the frequency of the ith intensity value
# the peak intensity p is identified by estimated its frequency fpeak as
    # fpeak = max(H-{f255})

def estimate_ah(bin_img):
    # getting the three most frequently occuring component heights
    image_invert = 255 - bin_img
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image_invert, connectivity=4)
    heights = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        heights.append(h)

    unique_heigths, counts = np.unique(heights, return_counts=True)
    sorted_indices = np.argsort(-counts)
    top_3_heights = [unique_heigths[i] for i in sorted_indices[:3]]
    # average of three heights == most frequent component height
    if len(top_3_heights) > 2:
        Ah = int(np.mean(top_3_heights))
    else:
        Ah = 0
    
    return Ah

def binarize(img, height, width, k=0.5 , R=128.0):
    mean = cv2.boxFilter(img, ddepth=-1, ksize=(height, width))

    sqr_mean = cv2.boxFilter(img**2, ddepth=-1, ksize=(height, width))
    std_dev = np.sqrt(sqr_mean - mean**2)

    T = mean * (1 + (k * ((std_dev/R) - 1)))

    binary = (img > T).astype(np.uint8) * 255

    return binary

input_labels_dir = ["dataset_split/train/labels", "dataset_split/val/labels", "dataset_split/test/labels"]
output_labels_dir = ["dataset_split_alteredbin/train/labels", "dataset_split_alteredbin/val/labels", "dataset_split_alteredbin/test/labels"]

# copy data.yaml
yaml_input = "dataset_split"
yaml_output = "dataset_split_alteredbin"
os.makedirs(yaml_output, exist_ok=True)

yaml_name = os.path.join(yaml_input, "data.yaml")
if (os.path.isfile(yaml_name)):
            if yaml_name.endswith('.yaml'):
                shutil.copy(yaml_name, yaml_output + "/")


input_dir = ["dataset_split/train/images", "dataset_split/val/images", "dataset_split/test/images"]
output_dir = ["dataset_split_alteredbin/train/images", "dataset_split_alteredbin/val/images", "dataset_split_alteredbin/test/images"]

for i, dir in enumerate(input_dir):
    os.makedirs(output_labels_dir[i], exist_ok=True)
    labels_filenames = [os.path.join(input_labels_dir[i], f) for f in os.listdir(input_labels_dir[i]) if os.path.isfile(os.path.join(input_labels_dir[i], f))]
    for label in labels_filenames:
         shutil.copy(label, output_labels_dir[i])

    os.makedirs(output_dir[i], exist_ok=True)
    image_filenames = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    image_filenames.sort()

    for img in image_filenames:
        print(dir + "/" + img)
        k = 0.5
        R = 128.0
        mu = 200

        image = cv2.imread(dir + "/" + img, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        h, w = image.shape

        while True:
            # iterative contrast stretching
            hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
            p = np.argmax(hist)

            if p >= mu:
                break
            
            image = np.where(image <= p, 255, image).astype(np.uint8)

        image = image.astype(np.float32)
        bin_img = binarize(image, h, w)

        # estimate dynamic window size
        Ah = estimate_ah(bin_img)
        WB = 2 * Ah + 1

        bin_image = bin_img.astype(np.float32)
        binary_final = binarize(bin_image, WB, WB)
        

        cv2.imwrite(output_dir[i] + "/" + img, binary_final)