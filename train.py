import os
import tensorflow as tf
import keras
import keras.preprocessing.image
import keras_retinanet
from keras_retinanet import models
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.models.resnet import resnet50_retinanet, ResNet50RetinaNet
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.callbacks import RedirectModel


#Paths GTSDB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOTS_DIR = os.path.join(BASE_DIR, 'snapshots')
DATASET_DIR = os.path.join(BASE_DIR, 'datasets', 'GTSDB', 'FullIJCNN2013')
GROUND_TRUTH = os.path.join(DATASET_DIR, "gt.txt")
TRAIN_GROUND_TRUTH = os.path.join(DATASET_DIR, "gt_train.csv")
TEST_GROUND_TRUTH = os.path.join(DATASET_DIR, "gt_test.csv")
CLASS_MAPPINGS = os.path.join(DATASET_DIR, "cm.csv")

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def create_models(num_classes, weights):
    #model = ResNet50RetinaNet(num_classes=num_classes, weights='imagenet', nms=False)

    inputs = keras.layers.Input(shape=(None, None, 3))

    #model = models.download_imagenet('resnet50')
    model = models.retinanet_backbone('resnet50')
    model.load_weights(weights, skip_mismatch=True)


    #model = resnet50_retinanet(num_classes=num_classes)
    training_model = model
    prediction_model = retinanet_bbox(model=model)

    #print(SNAPSHOTS_DIR)
    model.load_weights(os.path.join(SNAPSHOTS_DIR,weights))

    model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    #From imagenet
    #"https://github.com/fizyr/keras-models/releases/download/v0.0.1/ResNet-50-model.keras.h5"
    #.download_imagenet()

    return model, training_model, prediction_model

def create_generator():
    train_generator = CSVGenerator(
        TRAIN_GROUND_TRUTH,
        CLASS_MAPPINGS,
        batch_size=1,
    )

    return train_generator

def create_callbacks(model, training_model, prediction_model):
    callbacks = []

    # save the prediction model
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(SNAPSHOTS_DIR, 'custom_resnet50_{epoch:02d}.h5'),
        verbose=1
    )
    checkpoint = RedirectModel(checkpoint, prediction_model)
    callbacks.append(checkpoint)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1,
                                                     mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    callbacks.append(lr_scheduler)

    return callbacks

keras.backend.tensorflow_backend.set_session(get_session())
train_generator = create_generator()
model, training_model, prediction_model = create_models(num_classes=train_generator.num_classes(), weights='new_resnet50_csv_16.h5')
callbacks = create_callbacks(model, training_model, prediction_model)

# start training
training_model.fit_generator(
    generator = train_generator,
    steps_per_epoch = 850,
    epochs = 5,
    verbose = 1,
    callbacks = callbacks,
)