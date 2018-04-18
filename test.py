#This is pretty much just directly modified from: https://github.com/fizyr/keras-retinanet

# import keras
import keras

# import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
model_path = os.path.join('snapshots', 'model_ny.h5')

# load retinanet model
model = keras.models.load_model(model_path, custom_objects=custom_objects)


for imagename in [f for f in os.listdir("c:\\Users\\Mads\\Desktop\\dl\\FullIJCNN2013_converted\\") if os.path.splitext(f)[1] == ".jpg"]:
    # load image
    #image = read_image_bgr(os.path.join("c:\\Users\\Mads\\Desktop\\dl\\FullIJCNN2013_converted","00037".jpg"))
    image = read_image_bgr(os.path.join("c:\\Users\\Mads\\Desktop\\dl\\FullIJCNN2013_converted\\",imagename))
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        print("Score for image {}.jpg : {}".format(imagename, score))
        if score < 0.4:
            break
        
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.3f}".format(label, score)
        draw_caption(draw, b, caption)
        
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()