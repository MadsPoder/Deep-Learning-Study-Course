import keras
import numpy as np
import matplotlib.pyplot as plt
from keras_retinanet.utils.eval import evaluate

class save_weights_vis(keras.callbacks.Callback):
    def __init__(self, path, **kwargs):
        self.path = path
        super(self.__class__, self).__init__(**kwargs)
        
    def on_epoch_end(self, epoch, logs={}):      
        path = self.path.format(epoch + 1)
        fig = self.__draw_fig()
        print("Saving visualised weights of conv1 to file: {}".format(path))
        fig.savefig(path)
        plt.close(fig)
        
    def __draw_fig(self):
        #Get weights from conv1 layer
        weights = np.asarray(self.model.get_layer('conv1').get_weights())
        fig, ax = plt.subplots(figsize=(5, 5),nrows=8, ncols=8)
        for i, ax in enumerate(ax.reshape(-1)):
            imgArray = weights[0,:,:,:,i]
            # unity-based normalization https://datascience.stackexchange.com/a/5888
            imgArray = (imgArray - np.min(imgArray)) / (np.max(imgArray) - np.min(imgArray))
            ax.imshow(imgArray)
            ax.axis('off')
        return fig

#MODIFIED FROM: https://github.com/fizyr/keras-retinanet/blob/93618be71a23f12157497f7716e3e32a385e6680/keras_retinanet/callbacks/eval.py
class Evaluate(keras.callbacks.Callback):
    def __init__(self, generator, iou_threshold=0.5, score_threshold=0.05, max_detections=100, save_path=None, tensorboard=None, verbose=1, train=True):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.
        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
            tensorboard     : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            verbose         : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.verbose         = verbose
        self.train           = train

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        # run evaluation
        average_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        self.mean_ap = sum(average_precisions.values()) / len(average_precisions)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.mean_ap
            summary_value.tag = "mAP_train" if self.train else "mAP_test"
            self.tensorboard.writer.add_summary(summary, epoch)

        if self.verbose == 1:
            for label, average_precision in average_precisions.items():
                print(self.generator.label_to_name(label), '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(self.mean_ap))