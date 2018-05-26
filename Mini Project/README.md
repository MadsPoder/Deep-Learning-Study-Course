## Structure
### Notebooks
**annotate_video.ipynb** Prediction and annotation of video sequence.

**graphs.ipynb** Creation of Training/Validation accuracy graphs.

**[predict_test_set.ipynb](Deep-Learning-Study-Course/Mini Project/predict_test_set.ipynb)** Calculation of average scores per class.

**predict.ipynb** Annotation and prediction of single image.

**training.ipynb** Training script.

**visualisation.ipynb** Playground script for various visualisation techniques tried.

### Other files
**callbacks.py** Saving of weights, generation of mAP_train/mAP_Test.

**paths.py** Paths to files, DRY.

**shared.py** TF session and enums used.

## Prerequisites
1. Install Python 3.6.x
2. Install virtualenv
3. Clone the repository
4. Download the GTSDB dataset from [here](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)
5. Prepare the ground truths using *setup.ipynb*
6. Install [Keras RetinaNet](https://github.com/fizyr/keras-retinanet)

### Setup Windows GPU
(As of 15/04-18 pip version of tensorflow-gpu (1.7.0) only supports **CUDA 9.0**)
1. First install CUDA toolkit 9.0
2. Install cuDNN 7.0 for 9.0 (https://developer.nvidia.com/cudnn)

```
virtualenv venvgpu
venvgpu\Scripts\activate
pip install -r requirements_gpu.txt
```

### Setup Windows CPU
```
virtualenv venv
venv\Scripts\activate
pip install -r requirements_cpu.txt
```

### Setup OSX (CPU)
```
virtualenv -p python3 venv
. venv/bin/activate
pip install -r requirements_cpu.txt
```