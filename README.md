# Deep-Learning-Study-Course
Mandatory miniproject for the study course in deep learning at Aarhus University spring 2018

## Introduction
The ultimate goal of the project is to identify road signs in Denmark.

## Prerequisites
1. Clone the repository
2. Download the GTSDB dataset from [here](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)
3. Convert the dataset using *prepare_data.py*
4. Install [Keras RetinaNet](https://github.com/fizyr/keras-retinanet)


### Setup Windows GPU
(As of 15/04-18 pip version of tensorflow only supports **CUDA 9.0**)
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