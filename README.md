# Hyperface
Hyperface Architecture and train pipeline for [Hyperface](https://arxiv.org/pdf/1603.01249.pdf) paper in Keras with Tensorflow backend
For now only AlexNet based architecture is implemented.
### Train on sample data
1. Change the sample data path in ```config.py```. The data format should be numpy serializable 
and have the following format
For example, the sample data will be a numpy file. It will be a shape of ( X, 6 ) where X is the number of data points.
The content of each element will be
    1. Image (numpy array)
    2. List which has [ 1, 1 ] if the given image is face or [ 0, 0 ] if the
given image is non-face
    3. List of normalized facial coordinates [ 0.2536, 0.7890, …….. ]
    4. Visibility array [ 0, 0, 0, 1, 1, 1 …... ]
    5. Pose array [ 0.24304312, -0.42484489, -0.04113968 ]
    6. Gender array- [ 1, 0 ] for male and [ 0, 1 ] for female

2.  Change other parameters for training in ```config.py```.
3.  Start the training with

    ```$ python train.py```
    
### Pakages required
* Python 2.7 or Python3.7 (tested on both)
* tensorflow
* keras
* numpy
* Opencv (for image resizing)
