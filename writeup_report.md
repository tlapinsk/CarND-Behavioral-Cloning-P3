**Behavioral Cloning Writeup Report** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

## Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 showing the car lapping around track 1

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

1. Choosing an Architecture

My model consists of the Nvidia architecture as seen below

![alt text](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "Nvidia Convolutional Neural Network")

2. Reducing Overfitting

This project went through many phases of testing (both on the model and on gathering data). Originally, I began tinkering with the model by adding Dropout layers and changing from RELU to ELU. This was at the advice of many people in the forums. 

But, this approach actually did not yield great results for me. The model was trained on different data sets to ensure that the model was not overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. See below:

	del samples[0]
	train_samples, validation_samples = train_test_split(samples, test_size=0.2)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

3. Model Parameter Tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114).

4. Appropriate Training Data

I recorded many data sets (3 in fact) before I starting yielding great results. Recording, re-recording, and deleting images was probably the most time consuming part of this project. And in fact, it could have been a lot easier had I stumbled upon Kevin La Ra's [post](https://discussions.udacity.com/t/help-stuck-just-cant-get-a-full-lap-on-the-track/240144/6?u=tim.lapinskas) earlier on.

His advice to use 2 recordings (less is more) ended up being some of the best advice that I received. In addition Mohan's [blog post](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff) led me to start training the model purely with the Udacity data set from the start (found [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)).

I used a combination of Udacity's data set along with my own recovery data to get my car to finally lap around the track. It taught me a great deal about the importance of compiling excellent data for any deep learning project. It also seems to be one of the most (if not the most) time consuming part of deriving useful deep learning networks.

My recovery data was soley one lap of left and right side to center recording. There are definitely still some improvements that could be made to my recovery data - most likely which could be fixed with a joystick or steering wheel input.

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

1. Solution Design Approach

I didn't mess around much with the Nvidia architecture due to its usefulness and comparable nature of this project. This proved fruitful once I had recorded adequate training data.

Although, prior to recording useful data, I did attempt to employ a variety of pre-processing strategies. They were mainly derived from Mohan's [blog post](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff) and pniemczky's [forum post](https://discussions.udacity.com/t/still-having-some-issues-with-project-behavioral-cloning-any-advice/234354/15).

While it was interesting and also very useful to see the complex pre-processing strategies that other students used, it didn't help me early on in the project.

Moving on, I found renee's [https://discussions.udacity.com/t/filtering-out-70-of-the-images-that-are-straight/353824] to be VERY useful in terms of filtering out center images to gain a more even distribution overall of steering angles. I ended up implementing this into my final architecture, which increased performance by a wide margin.

In the end, simplifying my model ended up being the way to go. Huge shoutout to [Kevin La Ra](https://discussions.udacity.com/u/Kevin_La_Ra) and [Subodh Malgonde](https://discussions.udacity.com/u/subodh.malgonde) for hammering this point home across multiple forums posts.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

As noted above, I employed the Nvidia architecture

Here is a visualization of the architecture:

![alt text](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "Nvidia Convolutional Neural Network")

3. Creation of the Training Set & Training Process

To capture good driving behavior, I settled on starting with the Udacity data set ([found here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)). Center lane xamples of the data set found below:



I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the edges of the track. Check out a few images below:

![alt text](https://github.com/tlapinsk/CarND-Behavioral-Cloning-P3/blob/master/writeup_images/center_2017_09_10_13_10_01_376.jpg?raw=true "Center")
![alt text](https://github.com/tlapinsk/CarND-Behavioral-Cloning-P3/blob/master/writeup_images/left_2017_09_10_13_10_01_376.jpg?raw=true "Left")
![alt text](https://github.com/tlapinsk/CarND-Behavioral-Cloning-P3/blob/master/writeup_images/right_2017_09_10_13_10_01_376.jpg?raw=true "Right")

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Reflection
- It would be more interesting to come back to this project at some point and see if a combination of my now adequate training data and these pre-processing techniques would make the car perform even better.
