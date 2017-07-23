# Couldn't get this to show up correctly, so I creaded "writeup.pdf"  I'd appreciate some tips on how to get this markdown file to show up correctly.

#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image0]: ./examples/visualize0.png "Sample Image 0"
[image1]: ./examples/visualize1.png "Sample Image 1"
[image2]: ./examples/visualize2.png "Sample Image 2"
[image3]: ./examples/visualize3.png "Sample Image 3"
[image4]: ./examples/visualize4.png "Sample Image 4"
[image5]: ./examples/visualize5.png "Sample Image 5"
[image6]: ./examples/visualize6.png "Sample Image 6"
[image7]: ./examples/visualize7.png "Sample Image 7"
[image8]: ./examples/visualize8.png "Sample Image 8"
[image9]: ./examples/visualize9.png "Sample Image 9"
[image10]: ./downloaded-images/download (1).jpg "New Image 1"
[image11]: ./downloaded-images/images (1).jpg "New Image 2"
[image12]: ./downloaded-images/images (2).jpg "New Image 3"
[image13]: ./downloaded-images/images (3).jpg "New Image 4"
[image14]: ./downloaded-images/images.jpg "New Image 5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jsngithub/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is plot of 10 randomly selected traffic sign images

![alt text][image0]
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to normalize the images to zero mean and standard deviation of 1.  I did this so that images with different brightness and contrast would be treated in the same manner.  I decided to keep the images as color images becauce I thought color informatinon may be important in distinguishing the different traffic signs.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a slight tweak of the standard LeNet model used in class.  Since color images are used, the depth of the convolutional layers were slightly increased and the number of neurons in the fully connected layers increased slightly.

| Layer         		|     Description	        							| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   		
| Convolution 5x5		| 1x1 stride, valid padding, outputs 28x28x6 |
| ReLU				|				|
| Max pool				| 2x2 stride, valid padding, outputs 14x14x6 |
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x16 |
| Relu | |
| Max pool 			| 2x2 stride, valid padding, outputs 5x5x16 |
| flatten				| |
| Fully connected layer	| outputs: 120 |
| ReLU | |
| Dropout | |
| Fully connected layer    | outputs: 84 |
| Relu | |
| Dropout | |
| Fully connected layer	| outputs: 43 |


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To calculate the loss function, I first used the softmax cross entropy function for the logits output and one-hot encoded training output.  I then calculate the loss by summing the mean of the cross enropy and the produt of sum of the weights times a weight_cost(regularization).  To train the model, I used an AdamOptimizer.



####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.935
* test set accuracy of 0.933

I first started out with the default parameters carried over from the LeNet model lab.  The LeNet model had performed well on the number classification problem so it was reasonable to believe that it may perform well in the traffic sign classification.  The plan was to see how this architecture preformed and to develop further plans from there.  This resulted in validation accuracy of 0.910.  

I observed oscillation in the validation accuracy values between epochs.  Based on this, I thought that perhaps that model is overiftting the training data.  I introduced dropout and regularization in an effort to   With a keep rate of 0.7 and and weight_cost of 0.002, the validation accuracy increased to 0.935, with upward trending accuracy in all epochs but one.

While some hyperparemeter tuning here and increasing the depth/size of the neural network could potentially increase the accuracy, it was seen as unecessary due to a high accuracy rate.  High (and similar) accuracy level on the validation and traing show that the model is working well.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image10]
![alt text][image11]
![alt text][image12] 
![alt text][image13]
![alt text][image14]

The first image might be difficult to classify because the image is not centered and is a bit crooked.  The third one may be hard to classify since it is also crooked and occupies a fairly small portion of the image.  The fifth one may be difficult to classfy since the sign also occupies a fairly small portion of the entire frame.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection 		| Right-of-way at the next intersection						| 
| General caution     			| General caution     |
| Turn right ahead					| Turn right ahead|	
| Road work	      		| Road work	|
| Stop sign			| Priority road		|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

In all five images the model is not very certain about the predictions.  As a matter of fact, the highest probability for the correct prediction is only 29.7%.  On the fifth image where the prediction is wrong, the probability of it being a stop sign (correct answer) is not one of the top five softmax probabilities.

Image 1:

| Probability         	|     Prediction	| 
|:---------------------:|:---------------------------------------------:| 
| .27         			| Right-of-way at the next intersection |
| .18     				| Beware of ice/snow			|
| .09					| Pedestrians			|
| .09	      			| Children crossing		|
| .04				    	| Double curve 	|


Image 2:

| Probability         	|     Prediction	| 
|:---------------------:|:---------------------------------------------:| 
| .30         			| General caution   		| 
| .20     				| Traffic signals 			|
| .15					| Pedestrians		|
| .08	      			| Right-of-way at the next intersection		|
| .07				    	| Go straight or left	|

Image 3:

| Probability         	|     Prediction	| 
|:---------------------:|:---------------------------------------------:| 
| .08         			| Turn right ahead   		| 
| .07     				| Ahead only 			|
| .06					| Roundabout mandatory			|
| .05	      			| Turn left ahead		|
| .02				    	| Slippery Road 	|

Image 4:

| Probability         	|     Prediction	| 
|:---------------------:|:---------------------------------------------:| 
| .16         			| Road work   		| 
| .14     				| Dangerous curve to the right 			|
| .12					| Pedestrians			|
| .09	      			| Right-of-way at the next intersection		|
| .08				    	| Beware of ice/snow 	|

Image 5:

| Probability         	|     Prediction	| 
|:---------------------:|:---------------------------------------------:| 
| .38         			| Priority road   		| 
| .17     				| No passing for vehicles over 3.5 metric tons 			|
| .16					| Right-of-way at the next intersection			|
| .13	      			| Double curve		|
| .08				    	| End of no passing by vehicles over 3.5 metric tons 	|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


