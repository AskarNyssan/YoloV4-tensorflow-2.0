# Scaled YoloV4-tensorflow-2.0

## 1. Data
The model uses TF records as data source for training, so at first you need to generate TF records. Script tfrecords.py can be used to generate such data.

## 2. Training
After preparing necessary dataset, you could train Yolov4 model:
- python train.py

## 3. Yolov4 model.
- CSPDarknet53 backbone with Mish activations,
- SPP Neck,
- Training loop with YOLOv3 loss.
Also the whole Yolov4 architecture can be found in Schema scaled yolov4.xls
