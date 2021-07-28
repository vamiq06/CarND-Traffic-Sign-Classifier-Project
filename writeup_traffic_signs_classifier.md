# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualization.jpg "Visualization"
[image2]: ./normalization.jpg "Normalization"
[image4]: ./1.JPG "Traffic Sign 1"
[image5]: ./2.JPG "Traffic Sign 2"
[image6]: ./3.JPG "Traffic Sign 3"
[image7]: ./4.JPG "Traffic Sign 4"
[image8]: ./5.JPG "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Here is a link to my [project code](https://github.com/vamiq06/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 
I used the pickle library to import the training, testing and validation datasets from the downloaded p-files. Then divided the databases into image data(X_train, X_valid, X_test) and lable data(y_train, y_valid, y_test).

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 12630 images
* The size of test set is 4410 images
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

I was provided with a glossary of the sign names and their corresponding label in signnames.csv.
Using pandas library, I was able to import the signnames.csv file. I then visualized 10 images from the training set, chosen at random and labeled them using the sign names corresponding to the labels.

The following image is the visualization picture of the 10 images.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing the images
I normalized the image data because that centers the image data and it helps in classifying the image. The normalized images are represented here.
![alt text][image2]

#### 2. Description of the final model.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x26 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x26 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| output 400 features 							|
| Fully connected		| output 200 features 							|
| Fully connected		| output 84 features 							|
| Softmax				| output 43 classes								|



#### 3. Training of the model. 
My model consisted of 49 epochs and the batch size was 100 images per batch.
The hyperparameters for the truncated normal randomizer are mean = 0 and standard deviation = 0.1.
I used a Adam optimizer to train my model

#### 4. Training results

My final model results were:
* validation set accuracy of 94.2 
* test set accuracy of 94.8

I chose an iterative approach starting from the basic lenet architecture I had used in the class lectures.
This model itself was giving an 85% accuracy on the validation set.
The following parameters were available to be changed.
1. The features in the weights and bias
2. Hyper parameters mu and sigma in the truncated normal application function
3. Number of Epochs
4. Batch size
5. kernal shape in the convolution operation.

First I increased the number of epochs to 50 to see if the accuracy increases from the initial 85%. The accuracy increased to 89%

Then I tried changing the hyper parameters and the accuracy fell as I increased and decreased the hyper parameters. Hence, I used the original values of mu = 0 and sigma = 0.1 in my remaining models.

Then I tried reducing batch size from 128 to 100 and did not see a huge increment in the accuracy. The accuracy was still in the range of 89% and 90%. My processing time had been increased so I set the batch size to 100.

I changed the feature weights and bias by using 26 filters instead of original 6 filters in the first convilution. The accuracy jumped to 91.5% - 92%

Then I increased the weights inthe first fully connected layer to get an output of 200 weights and the accuracy increased to 94.2%

I did not do any iterations on the kernel size

### Testing the Model on New Images

#### 1. Choosing the 5 images from the interner

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

These images were preporcessed so that the size is compressed to (32,32,3) using opencv library and interpolation using cv2.resize() operation.
Using numpy I stacked these images into an array img_stack
The labels were assigned in an array label_img_stack
After normalizing the img_stack images, I ran them through the model to predict the class

#### 2. Discussing the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

The 5 images true lables and the predictions are expressed in the table below.

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children Crossing		| Children Crossing								| 
| Roadwork  			| Roadwork  									|
| Go straight or right	| Go straight or right							|
| 80 km/h				| 80 km/h						 				|
| 30 km/h				| 80 km/h										|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.8%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

The first image contains a children crossing sign. The model is has highest probability that this is a children crossing sign(probability of 32%) the next best probability match is 19% for Dangerous curve to the right. The second, third and fourth predictions are in a similar range.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 32         			| Children crossing   							| 
| 19     				| Dangerous curve to the right					|
| 18					| Beware of ice/snow							|
| 17	      			| Right-of-way at the next intersection			|
| 08				    | Bicycles crossing 							|


The second image contains a Road work sign. The model is has highest probability that this is a Road work sign(probability of 53%) the next best probability match is 27% for Dangerous curve to the left. The second and third predictions are in a similar range.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 53         			| Road Work										| 
| 27     				| Dangerous curve to the left					|
| 23					| Traffic signals								|
| 13	      			| Road narrows on the right						|
| 09				    | Beware of ice/snow							|

The third image contains a Go straight or right sign. The model is has highest probability that this is a Go straight or right sign(probability of 78%) the next best probability match is 59% for End of no passing. The second, third and fourth predictions are in a similar range.

The top five soft max probabilities were

| Probability 			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 78					| Go straight or right							| 
| 59  					| End of no passing								|
| 51					| Roundabout mandatory							|
| 45					| General caution								|
| 38					| keep right									|

The fourth image contains a Speed limit (80km/h) sign. The model is has highest probability that this is a Speed limit (80km/h) sign(probability of 34%) the next best probability match is 20% for Speed limit (30km/h). The second and third predictions are in a similar range but much behind the first prediction.

The top five soft max probabilities were

| Probability 			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 34					| Speed limit (80km/h)							| 
| 20  					| Speed limit (30km/h)							|
| 15					| No passing									|
| 04					| Speed limit (60km/h)							|
| 04					| Road Work										|

The fifth image contains a Speed limit (30km/h) sign. The model is has highest probability that this is a Speed limit (30km/h) sign(probability of 65%) the next best probability match is 59% for Speed limit (70km/h) which is near the first guess. The third, fourth and fifth predictions are in a similar range but much lower than the first two predictions.

The top five soft max probabilities were

| Probability 			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 65					| Speed limit (30km/h)							| 
| 59  					| Speed limit (70km/h)							|
| 15					| Speed limit (50km/h)							|
| 11					| Speed limit (80km/h)							|
| 11					| General caution								|
