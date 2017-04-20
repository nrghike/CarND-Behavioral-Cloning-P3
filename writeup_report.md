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

[image1]: ./write_up_data/nVidia_model.png "Model Visualization"
[image2]: ./write_up_data/Original_data_hist.png "Original Histogram"
[image3]: ./write_up_data/Processed_Hist.png "even distribution Histogram"
[image4]: ./write_up_data/Post_Bridge.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* My_Nvidia_Model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the project

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

After reading through some [literature](https://medium.com/self-driving-cars/6-different-end-to-end-neural-networks-f307fa2904a5), I decided to go ahead with the NVIDIA. The NVIDIA architecture was chosen for this project since it had been already used for a similar application.

Model consists of a convolution neural network with 5x5 filter sizes and depths between 32 and 128. This has been adopted from the [NVIDIA Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf):

![alt text][image1]


Below is the summary of my model. The model includes RELU layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer.

```python
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate), loss="mse", )
```


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting:
```python
model.add(Dropout(0.2))
```
The model was trained and validated on different data sets to ensure that the model was not overfitting. The samples were split into training (85%) and validation (15%) sets randomly:
```python
images_train, images_validation, angles_train, angles_validation = train_test_split(
    images, angles, test_size=0.15, random_state=42)
```
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.The model was compiled using the Mean Square Error.


####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Training Strategy


####1. Creation of the Training Set & Training Process

Data was collected while keeping the vehicle in the center. The left, center and right camera were used. A number of samples were also collected such that the vehicle vehicle was recovering. Additional data was collected at turns where the vehicle was failing to keep in the lane.

<p align="left">
  <img src="write_up_images/left.jpg" alt="Left" height="80"/>
</p>
<p align="center">
  <img src="write_up_images/center.jpg" alt="Center" height="80"/>
</p>
<p align="right">
  <img src="write_up_images/right.jpg" alt="Right" height="80"/>
</p>



It was observed that many of the images had zero steering angle associated with them. Hence even if the vehicle was headed towards the outside of the road, it did not learn to recover because it would predict the angle as 0.

![alt text][image2]


I increased the number samples by data augumentation. The technique I used was to remove some of the data with 0 steering angle so that we have even distribution. This gave me an even distribution as shown below:

![alt text][image3]


I played around with some augumentation techniques as well.
1. To flip the Image. This would take care of some missing right turn data.
2. To tint the Image. This would compensate for lighting conditions.

Jitter helped generalized the model significantly. It may be one of the reason the vehicle was able to maneuver the turn after the bridge.
```python

def jitter_image(path, steering):
    image = cv2.imread(path.strip())
    rows, cols, _ = image.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))

    # Convert the image into mulitdimensional matrix of float values (normally int which messes up our division).
    image = np.array(image, np.float32)

    # Crop Image
    image = image[35:135, :]
    image = scipy.misc.imresize(image, (66,200))

    return image, steering
```


#### Training Process

I used Batch Generators. Generators helped control the number of samples sent through a batch. This enabled learning on GPUs with limited memory.
As I was using AWS, I choose a batch size of 64. If it would have been my PC, I would have preferred a batch size of 256 because of GPU Limitation.


#### Testing process

I encountered a couple of problems during testing.


1. The vehicle kept going towrds left side in my first try and it went off road after the bridge when we lost Lane markings.

![alt text][image4]

This issue was relevant on the brdge as well. I resolved it by collecting more data at these patches of track.


2. Vehicle kept steering towards left.

This was easily resolved as stated earlier by balancing the dataset.


