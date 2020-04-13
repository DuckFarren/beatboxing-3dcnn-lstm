# BeatBoxing - Low-latency Real time Gesture Recognition
This project is a boxing minigame implemented with a self-trained deep learning model (3DCNN+LSTM) using OpenCV, Keras/Tensorflow libraries in Python

### Preprocessing
Due to the limited amount of data we collected, we use the below methods to increase the size of the dataset:
1. Video Subsampling (e.g taking odd number frames/even number frames)
2. Video Augmentation (Credits to https://github.com/okankop/vidaug)
![](images/vidaug.JPG)

The model is taking 24frames for prediction. This number can be reduced in the training process.

### Training Result
After training for 50 epochs, we achieved >95% of validation accuracy.
![](images/model_train.JPG)

### Demo
![](images/demo.gif)


### Presentation
[4min presentation video](https://drive.google.com/open?id=1AdlkjEE0CZe0zR6lMfsjTp1nCaj8DZn2)

[Presentation slides](https://drive.google.com/open?id=1Db7DqxLoZQXVJS-kL38dzX-DNYAHqeig)

