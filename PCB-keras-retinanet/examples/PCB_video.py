from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import numpy as np
import time
import cv2
import os

# adjust this to point to your downloaded/trained model
model_path = os.path.join('..', 'snapshots', 'resnet50_csv_38.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)
#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Mousebite',1: 'Copper',2: 'Short',3: 'Open',4: 'Pin-Hole',5: 'Spur'}


vid = cv2.VideoCapture('./PCB_60.avi')

while(vid.isOpened()):
    return_value, frame = vid.read()

    draw = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # preprocess image for network
    image = preprocess_image(draw)
    image, scale = resize_image(image)
    
#     image = yolo.detect_image(image)
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    t = time.time() - start
    boxes /= scale
    
        # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores aqre sorted so we can break
        if score < 0.3:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

        
        cv2.imshow('frame',draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
vid.release()
cv2.destroyAllWindows()


