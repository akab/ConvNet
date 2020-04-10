### Dependencies 
Required:
```
+ CUDA v10.1
+ cuDNN v7.5.0 for CUDA 10.1
+ Python 3.7
+ tensorflow-gpu
+ keras
+ keras-applications
+ keras-preprocess
```

Additional:
```
+ numpy
```

### HowTo install tensorflow with GPU support on Windows 7

+ Download and install CUDA Toolkit v10.1 from: https://developer.nvidia.com/cuda-downloads

+ Download and install cuDNN v7.5.0 from (registration needed): https://developer.nvidia.com/cudnn

+ Set environmental variables:
	+ CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
	+ PATH += C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;<br>
	          C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\libnvvp

+ Install Anaconda from: https://www.anaconda.com/download/

+ Create conda environment: 
```
conda create -n tf-gpu python=3.6
```

+ Activate the environment ('activate tensorflow-gpu') and install tensorflow-gpu with: 
```
conda install tensorflow-gpu keras-gpu
```

+ To test installation, launch a python shell and try:
```
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```
In the list of available devices should be present at least one 'gpu/gpu_device'.<br>
More informations availabe at: https://www.tensorflow.org/install/gpu