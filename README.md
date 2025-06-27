This repository contains the dataset, scripts and models used in the bachelor thesis "Optimizing YOLOv8m for the Detection of Layout Elements in Early Modern British Literature" by Ross Geurts for the BSc Kunstmatige Intelligentie at the University of Amsterdam. 
What follows is a brief description of the documents in this repo and how to use them.

__annotated_dataset__

This folder contains the dataset split into a test (100 images) and train (460 images) set, with each image having a corresponding text file containing the location and label of each bounding box in that image. These labels are in YOLOv8 (txt) format.

__unannotated_dataset__

This folder contains all images without the labels folder. 

__scripts__
- _test_train_val_splits.py_ : When the dataset only contains one folder with all images, use this script to create a 80/10/10 training, validation and testing split.
- _train_val_split.py_ : When using a datset which already has a split between training and test, use this script to split the training set into training (80%) and validation sets (20%).
- _binarization.py_ : This script details the binarization pipeline, which automatically copies all labels to the new dataset folder it creates in the process. No alteration is necessary except changing the path of each folder to the relative location on your personal computer.
- _gaussian_blur.py_ : This script applies Gaussian blurring to the training set and creates a new dataset in the process; the validation and test set, along with the labels of all splits will be copied as well.
- _training_yolov8.py_ : This script indicates the training of YOLOv8. The only necessary part is to alter the parameters in __parameters.yaml__ to the preferred parameters and changing the dataset path to your preferred dataset.

__models__

The __yolov8m.pt__ model indicates the standard YOLOv8m model, while each different model contains the weights of one of the models highlighted in the thesis, with the names corresponding to the abbreviations used for these models.

If there are any questions about the usage of this repository, inquiries can be sent to rmgeurtd@gmail.com.
