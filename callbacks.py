import keras
import numpy as np
import matplotlib.pyplot as plt

class save_weights_vis(keras.callbacks.Callback):
    def __init__(self, path, **kwargs):
        self.path = path
        super(self.__class__, self).__init__(**kwargs)
        
    def on_epoch_end(self, epoch, logs={}):      
        path = self.path.format(epoch + 1)
        fig = self.__draw_fig()
        print("Saving visualised weights of conv1 to file: {}".format(path))
        fig.savefig(path)
    
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