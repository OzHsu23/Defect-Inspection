# Defect-Inspection
In Defect-Inspection,I tried the two Deep learning methods to defect inspection(PCB & AOI).Moreover, I also imitated the traditional method to defect inspection by OpenCV.
The outcomes and demo is below:

RetinaNet  Demo

In RetinaNet Demo follow from:https://github.com/fizyr/keras-retinanet , thanks for the author.
The difference is I train the model in the PCB & AOI dataset and add the detect vedio part in example/PCB_video.py

In CPU:

AOI DataSet mAP:96.24% Time:2.620s/per img

PCB DataSet mAP:95.12% Time:3.168s/per img


Yolo3 Demo

In Yolo3 Demo follow from:https://github.com/qqwweee/keras-yolo3, thanks for the author.
Also, I train the model in the PCB & AOI dataset.

In CPU:

AOI DataSet mAP:93.79% Time:1.077s/per img

PCB DataSet mAP:86.01% Time:0.995s/per img

Conculsion:Yolo3(fast), RetinaNet(precise)
