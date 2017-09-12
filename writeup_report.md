# Behavioral Cloning Writeup Report

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
* Have fun!

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

## Files Submitted & Code Quality

Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing the car lapping around track 1

#### Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

#### Choosing an Architecture

My model consists of the Nvidia architecture as seen below:

![alt text](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "Nvidia Convolutional Neural Network")

#### Reducing Overfitting

This project went through many phases of testing (both on the model and on gathering data). Originally, I began tinkering with the model by adding Dropout layers and changing from RELU to ELU. This was at the advice of many people in the forums. 

But, this approach actually did not yield great results for me so I scrapped those strategies. The model was trained and validated on different data sets to ensure that the model was not overfitting. See below:

	del samples[0]
	train_samples, validation_samples = train_test_split(samples, test_size=0.2)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model Parameter Tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114).

#### Appropriate Training Data

I recorded many data sets (3 in fact) before I starting yielding great results. Recording, re-recording, and deleting images was probably the most time consuming part of this project. And in fact, it could have been a lot easier had I stumbled upon Kevin La Ra's [post](https://discussions.udacity.com/t/help-stuck-just-cant-get-a-full-lap-on-the-track/240144/6) earlier on.

His advice to use 2 recordings (less is more) ended up being some of the best advice that I received. In addition Mohan's [blog post](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff) led me to start training the model purely with the Udacity data set from the start (found [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)).

I used a combination of Udacity's data set along with my own recovery data to get my car to finally lap around the track. It taught me a great deal about the importance of compiling excellent data for any deep learning project. It also seems to be one of the most (if not the most) time consuming parts of deriving useful deep learning networks.

My recovery data was soley one lap of left and right side to center recording. There are definitely still some improvements that could be made to my recovery data - most likely which could be fixed with a joystick or steering wheel input.

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

#### Solution Design Approach

I didn't mess around much with the Nvidia architecture due to its usefulness and the comparable nature of the projects. This proved fruitful once I had recorded adequate training data.

Although, prior to recording useful data, I did attempt to employ a variety of pre-processing strategies. They were mainly derived from Mohan's [blog post](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff) and pniemczky's [forum post](https://discussions.udacity.com/t/still-having-some-issues-with-project-behavioral-cloning-any-advice/234354/15).

While it was interesting and also very useful to see the complex pre-processing strategies that other students used, it didn't help me early on in the project.

Moving on, I found renee's [post](https://discussions.udacity.com/t/filtering-out-70-of-the-images-that-are-straight/353824) to be VERY useful in terms of filtering out center images to gain a more even distribution overall of steering angles. I ended up implementing this into my final architecture, which increased performance by a wide margin. See code example below:

	...

	images = []
	angles = []
	adjustment = 0.15
	for batch_sample in batch_samples:
	    
	    # Read in center, left, and right images
	    center_image = '/home/carnd/data/IMG/'+batch_sample[0].split('/')[-1]
	    left_image = '/home/carnd/data/IMG/'+batch_sample[1].split('/')[-1]
	    right_image = '/home/carnd/data/IMG/'+batch_sample[2].split('/')[-1]
	    
	    # Grab images and associated angles
	    center_image = cv2.imread(center_image)
	    center_angle = float(batch_sample[3])
	    left_image = cv2.imread(left_image)
	    left_angle = float(batch_sample[3]) + adjustment
	    right_image = cv2.imread(right_image)
	    right_angle = float(batch_sample[3]) - adjustment
	    
	    # Extend images to lists
	    images.extend((center_image, left_image, right_image))
	    angles.extend((center_angle, left_angle, right_angle))
	    
	...

In the end, simplifying my model ended up being the way to go. Huge shoutout to [Kevin La Ra](https://discussions.udacity.com/u/Kevin_La_Ra) and [Subodh Malgonde](https://discussions.udacity.com/u/subodh.malgonde) for hammering this point home across multiple forums posts.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Final Model Architecture

As noted above, I employed the Nvidia architecture. A visualization of the architecture can be seen above.

#### Creation of the Training Set & Training Process

To capture good driving behavior, I settled on starting with the Udacity data set ([found here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)). Center lane examples of the data set found below:

![alt text](https://github.com/tlapinsk/CarND-Behavioral-Cloning-P3/blob/master/writeup_images/center_2016_12_01_13_30_48_287.jpg?raw=true "Udacity Center")
![alt text](https://github.com/tlapinsk/CarND-Behavioral-Cloning-P3/blob/master/writeup_images/center_2016_12_01_13_45_53_138.jpg?raw=true "Udacity Center")

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the edges of the track. Check out a few images below:

![alt text](https://github.com/tlapinsk/CarND-Behavioral-Cloning-P3/blob/master/writeup_images/center_2017_09_10_13_10_01_376.jpg?raw=true "Center")
![alt text](https://github.com/tlapinsk/CarND-Behavioral-Cloning-P3/blob/master/writeup_images/left_2017_09_10_13_10_01_376.jpg?raw=true "Left")
![alt text](https://github.com/tlapinsk/CarND-Behavioral-Cloning-P3/blob/master/writeup_images/right_2017_09_10_13_10_01_376.jpg?raw=true "Right")

To augment the data sat, I also flipped images and angles thinking that this would help the model generalize. 

After the collection process, I had 8,256 data points. I then preprocessed this data by employing filtering and cropping. This was noted earlier in this writeup and also in many of the forum posts that I came across. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I also created a generator, which proved very helpful. Early on in my testing, I ran into memory errors and Udacity's generator example helped immensely to efficiently train my model. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by my loss graph. Again, this was chosen in help by a [forum response](https://discussions.udacity.com/t/number-of-epochs/228034/5) by Subodh Malgonde. See below for a picture of my loss graph:

![alt text](https://github.com/tlapinsk/CarND-Behavioral-Cloning-P3/blob/master/writeup_images/Loss_Plot.png?raw=true "Loss Plot")

## Reflection
Overall, I would say that my car performs adequately, but that there is significant room for improvement. A few notes are below on where I believe I can further improve the performance of my car.

#### Performance Improvement Reflection
- Early on I deployed a variety of pre-processing strategies. My final architecture doesn't include these, but it would be interesting to add them in now to see if the car performs even better. Some notable techniques include: Gaussian blur, changing to YUV color space, changing X and Y translation, and brightening the images.
- Increasing speed. I did not attempt to increase the speed of the car in drive.py and would love to see how my car performs at higher speeds. 
- Attempting track 2. I did not attempt to run my car on track 2, and believe that it would perform very poorly since I did not collect any data from the track. My model, pre-processing stratagies, and data set would need to be significantly improved for decent performance on track 2. 

#### Miscellaneous Reflection
- I would love to employ other network architectures to see how they perform. VGG16 and ResNet are a couple that would be fun to test out in my opinion.
- The biggest takeaway for me in this project is how to approach a deep learning problem. By starting small (small data set, small model, etc.) you can prove that everything works (or overfits) and then move on to more complex processing or bigger nets from there. 
- Data data data. You are only as good as your data. This project taught me to complexities of recording excellent data and I can't image how difficult this process is in the field when you're driving real cars. State of the art systems must sift through mountains of data to pull out the data points that end up being useful.

## Other Resources
- [Discussion about how to approach this project](https://discussions.udacity.com/t/strange-behavior-of-trained-net-and-autonomous-mode-in-simulator/228667/14)
- [More discussion about how to approach this project](https://discussions.udacity.com/t/reason-why-some-architectures-do-horribly/230564/3)
- [Displaying deep learning history in Keras](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
- [Improving the model further](https://discussions.udacity.com/t/idea-to-further-improve-my-model/354787)
- [Recording recovery data](https://discussions.udacity.com/t/car-hugs-the-side-line/230703/6?u=subodh.malgonde)
- [Distribution of images](https://discussions.udacity.com/t/model-not-able-to-learn-some-portions-of-the-track/234995)
- [Vivek's blog post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)
- [Behavioral cloning cheat sheet](https://files.slack.com/files-pri/T2HQV035L-F50B85JSX/download/behavioral_cloning_cheatsheet_-_carnd__1_.pdf?pub_secret=7d8737aeeb)
- [Keras tutorial on image classification](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
