#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./write_up_imgs/center_image.jpg „Normal Image“
[image3]: ./write_up_imgs/center_left_recovery.jpg "Recovery Image"
[image4]: ./write_up_imgs/center_left_recovery1.jpg "Recovery Image"
[image5]: ./write_up_imgs/center_left_recovery2.jpg "Recovery Image"
[image6]: ./write_up_imgs/center_image_flipped.jpg „Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_udacity.h5 containing a trained convolution neural network using only udacity's dataset
* model_owndata.h5 containing a trained convolution neural network using my own dataset
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model_udacity.h5/ model_owndata.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 48 (model.py lines 91-97)

The model includes RELU layers to introduce nonlinearity (code line 91-108), and the data is cropped in the model using a Keras cropping layer (code line 89).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 101, 104).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 82, 115). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 111).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used udacity's data to train my model and I collected data myself. To get the absolute best out of udacity's and my dataset I incorporated left, right and center images! I added an offset to all angles from the left and right images to simulate a steering to the center of the image. Also the data were preprocessed by smoothing input angles (using rolling mean, moving average) to limit jumps in steering-degree. The images were also resized to 80 by 160 pixels for better training performance. The data were also augmented with ImageDataGenerator.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep it simple.

First I tried different approaches to accomplish a well trained model. I used a lot of layers and my model became to certain in a way, that it doesn’t knew what to do, when the situation wasn’t exactly as in the training data. So I used less convolutional layers and bigger layers as well as cropping the images. Filter size were reduced as well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that in some of my previous models I had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to include dropout layers (2 with 0.1 dropout).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track due to extreme steering angles and oscillation to improve the driving behavior in these cases, I modified the drive.py file to reduce extreme steering angles by toning down the output of the steering angles. I also modified the acceleration of the vehicle, so that it holds the speed around 12 mph and in cases were it drops below 12mph it increases acceleration.

At the end of the process, the vehicle is able to drive autonomously around track 1 without leaving the road. It’s also able to drive on track two (of the old simulator)! (I collected my training data in the old simulator, I hope that’s no problem. At first I didn’t even noticed that the simulator had changed!)

####2. Final Model Architecture

The final model architecture (model.py lines 87-113) consisted of a convolution neural network with the following layers and layer sizes: three 5x5 convolutional layers  each followed by max pooling layers. Border mode is ‚valid‘ and my activation is a relu activation. Dropouts were incorporated after the flattening and after the first fully connected layer.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to react in situations, where it isn’t on the ideal route. These images show what a recovery looks like starting from left to center :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles because the  first track is left-turn-dominant and this way the model has equally amounts of right and left turns. I actually used this method but in my own dataset I drove the track backwards, so that it wasn’t necessary anymore. Also I think this could be beneficial, because I don’t drive the same all the time and flipped images would be just the same but upside down, where as my backwards driving could give the model slightly different angles and images which would make the model more robust! For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 45,000 data points. I then preprocessed this data by resizing all images to 80 by 160 and cropping the hood to the car and the horizon of the images. And smoothing the angles by applying a rolling mean function on my angle data. Also I used left and right images and added offset to the right and left steering angles.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by low squared mean and bottoming out of the learning curve. I used an adam optimizer so that manually training the learning rate wasn't necessary.
