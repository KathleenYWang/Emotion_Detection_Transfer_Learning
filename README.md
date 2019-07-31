# Emotion Detection with Transfer Learning

### Overview
Accurately identifying human emotions is a challenging task even for humans. With the help of deep neural networks, significant progress has been made in training algorithms to identify emotions. Here, we explore using transfer learning to retrain and refine complex deep neural networks with pre-trained weights.

### Data
Extensive research has been done in this area and increasing amount of data have been used by researchers.
In this project, we trained our base model on the FER 2013 dataset and further evaluated the results agaisnt the SFEW dataset

### Tested Models
Two models were tested. The base line was the InceptionV3 model pre-trained on imagenet. This model is very large and only thanks to transfer learning, we can re-train this model using the few images available.  

However, we tried a smaller model called [Mini-Xception](https://github.com/oarriaga), originally proposed by Octavio Arriaga, Matias Valdenegro-Toro and Paul Plöger. This model is much faster to train and worked just as well. 

