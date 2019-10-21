# HackBJ Repository (HelloWorld)

![Team](https://github.com/dalsontws/HackBJ/blob/master/docs/team.jpg)

# Children Emotion detector

## We are HelloWorld!

We aim to help parents understand more about their children's emotions and mental well-being through seamless tracking of the emotions and FFT of the children. 

Powered by OpenCV and Deep Learning.

Special thanks to:

Reference: https://github.com/petercunha/Emotion.git, StackOverFlow, SegmentFault

## Installation and Execution

cd children-emotion
python3 fft.py 
python3 emotions.py 

## The fft is to track the Fast Fourier Transform (High FF)

Cry signals can be described by their features within two common domains: (1) the time domain and (2) frequency domain. From either of the mentioned domains, a number of significant characteristics can be extracted (Scherer, 1982). In this section, we describe the different domain features applied at different levels of our work.

1. Time-domain features
a. Intensity.
The intensity is also called the loudness, and it is related to the amplitude of the signal. It represents the amount of energy a sound has per unit area. The intensity is defined by the logarithmic measure of a signal section of length N in decibels as follows:

I=10  log (∑n=1Ns2(n)w(n)),

where w is a window function and s(n) is the amplitude of the signal.

Intensity is an essential feature widely used in different applications, such as music mood detection, and an accuracy rate of 99% is achieved, proving the good performance of intensity features.

This model is backed by the research paper by Scherer K. R. (1982). “ The assessment of vocal expression in infants and children,” in Measuring Emotions in Infants and Children.


## The emotions is a computer vision to track emotions


```

Install these dependencies with `pip3 install <module name>`
-	tensorflow
-	numpy
-	scipy
-	opencv-python
-	pillow
-	pandas
-	matplotlib
-	h5py
-	keras

Once the dependencies are installed, you can run the project.
`python3 emotions.py`
```

## To train new models for emotion classification

- Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Move the downloaded file to the datasets directory inside this repository.
- Untar the file:
`tar -xzf fer2013.tar`
- Download train_emotion_classifier.py from orriaga's repo [here](https://github.com/oarriaga/face_classification/blob/master/src/train_emotion_classifier.py)
- Run the train_emotion_classification.py file:
`python3 train_emotion_classifier.py`


## Deep Learning Model

The model used is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf) written by Octavio Arriaga, Paul G. Plöger, and Matias Valdenegro.

![Model](https://i.imgur.com/vr9yDaF.png?1)


## Credit

* Computer vision powered by OpenCV.
* Neural network scaffolding powered by Keras with Tensorflow.
* Convolutional Neural Network (CNN) deep learning architecture is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf).
* Pretrained Keras model and much of the OpenCV code provided by GitHub user [oarriaga](https://github.com/oarriaga).

# Future Improvement
## Baby crying will last more than 93ms (Factor these facts to get more accurate baby cry detection)


