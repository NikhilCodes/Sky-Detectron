# SKY Detectron
Python OpenCV and CNN implementation to mask sky pixels with white and rest of the region as black.

# Dependencies
 - tensorflow (2.x)
 - PyQt5
 - OpenCV

Make sure you have upgraded `pip` version to be able to install Tensorflow.

## Installing all dependencies
`$ pip install -U pip`

`$ pip install -r requirements.txt`

# Building tflite version of CNN model
Make sure you have `make` installed.
(Linux users can ignore as they already have `make` installed!)

Then Run, 

`make tflite`
