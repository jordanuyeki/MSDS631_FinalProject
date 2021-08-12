# Face Mask Detection Using Neural Networks

**By**: Matthew Hui and Jordan Uyeki

This repository contains our final project for our MSDS 631: Deep Learning course, which is the final course in our Masters in Data Science program at the University of San Francisco. 

## Our Goal 
For this project, we wanted to leverage various neural network models to classify whether or not an individual in a given image is wearing a face mask. With the ongoing pandemic and various mask mandate rules being enforced around the world, we feel that our project could have many practical implications. For example, in a more advanced application, our final model could be used to analyze surveillance camera footage snapshots and flag people who are potentially violating mask mandates.  

## The Data
The dataset we chose to work with is the [COVID Face Mask Detection Dataset](https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset) from Kaggle. This dataset contains over 1000 images, which each fall into two distinct categories: 
- The Mask Wearers:     
![mask-example](/images/mask.png)

- The Non Mask Wearers:     
![nonmask-example](/images/nonmask.jpg)

## Our Process

### Data Processing 
The main preprocessing step that we did involved resizing the images prior to modeling rather than doing it in our Dataset class as this is more efficient. We resized all of the images to 128x128 pixels for our Convolutional Neural Network model and to 224x224 pixels for our pretrained Mobilenet V2 and pretrained Resnet18 models. 

### Fitting and Comparing Candidate Models
To find the best model, we compared the accuracy of three candidate models: 

#### Convolutional Neural Network      
We used a Convolutional Neural Network created from scratch that contained 4 convolution layers and 2 linear layers. We used a Cross Entropy Loss Function with an Adam optimizer and trained the model with various learning rates and epochs, summarized as follows: 
 Epochs | Learning Rate | Training Accuracy | Validation Accuracy
------------ | ------------- | ------------- | -------------
5 | 0.01 | 0.500 | 0.500
5 | 0.001 | 0.890 | 0.879
20 | 0.0001 | 0.900 | 0.889

#### Pretrained Mobilenet V2
For our second candidate model, we used a pretrained Mobilenet V2 model from the torchvision module.  We used a Cross Entropy Loss Function with an Adam optimizer with a learning rate of 0.001. After training for 5 epochs, we attained a training accuracy of 0.980 and a validation accuracy of 0.950. 

#### Pretrained Resnet 18
For our last candidate model, we used a pretrained Resnet18 model from the torchvision module.  We used a Cross Entropy Loss Function with an Adam optimizer with a learning rate of 0.001. After training for 5 epochs, we attained a training accuracy of 0.972 and a validation accuracy of 0.948. 

Below is a table that summarizes our candidate models' performances
* | Training Accuracy | Validation Accuracy
------------ | ------------- | -------------
Convolutional Neural Network | 0.900 | 0.889
Pretrained Mobilenet V2 | 0.980 | 0.950
Pretrained Resnet 18 | 0.972 | 0.948

### Final Model
We chose the pretrained Mobilenet V2 model as our final model as it has the highest accuracy on the training and validation sets. For the final model, we combined the training and validation sets to train the model and tested it on the withheld test set which consisted of 100 images, 50 mask and 50 non-mask wearers. After training for 3 epochs, we got a final accuracy of 98 percent, with only 2 non-mask images being classified as mask images. 

## Future Directions 
This dataset was very clean and contained fairly straightforward images with the individuals being facing forward with minimal background distractions. This likely contributed to our high accuracies. In future iterations of this project, we would like to develop our model on more diverse data that consists of more noisy images to allow our model to be more robust and uphold its high accuracy on more realistic data. More diverse data would also give us an opportunity to explore the weaknesses in the model. These were the only two images from the test data that our model misclassified.         
![misclassified1](/images/misclassified1.png)
![misclassified2](/images/misclassified2.png)

It is difficult to make conclusions about our model weaknesses from just these two images. However with more diverse data (more misclassifications), we might be able to identify patterns and biases that the model is picking up on. 

Thank you for taking the time to read about our final project. We hope you learned something useful!


