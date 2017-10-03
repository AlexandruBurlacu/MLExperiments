# Hylang Machine Learning models

This directory contains machine learning example scripts translated
from Python to [Hylang](http://docs.hylang.org).

It was done just for the sake of learning, and because LISPs are sooo cool.

Currently there's a Scikit-Learn example inspired from [here](http://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py) and 3 Tensorflow examples.
The `mnist-simple` is based on [this](https://www.tensorflow.org/get_started/mnist/beginners), the other two are inspired from [here](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py) (`mnist-advanced`) and [here](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py) (`mnist-tensorboard`).

### Note
The `mnist-advanced.hy` isn't working for now, due to an issue with early return from a function in Hylang.
