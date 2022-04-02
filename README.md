# facemask_detection_python

The aim of project is to distinguish whether the faces that the camera sees and focuses on are wearing masks or not.
In this project machine learning and its algorithms  was  implemented to train and test the proper dataset .
While developing this project with python, used many libraries for machine learning, image processing and mathematical analysis.
In the first phase of the project, researched these libraries and found the features and algorithms ı should focus on.
Some of the algorithms and libraries used:

⦁	From tensorflow.keras.application import MoblieNetV2
MobileNet is also a CNN vision model.
The CNN algorithm, which makes image classification, was used to decide whether there is a mask on the face or not in our project.
⦁	From tensorflow.keras.preprocessing.image import ImageDataGenerator 
Image data augmentation is a technique that can be used to artificially expand the size of a training dataset by creating modified versions of images in the dataset.

The Keras deep learning neural network library provides the capability to fit models using image data augmentation via the ImageDataGenerator.

⦁	from tensorflow.keras.optimizers import Adam 

Optimizers updates with gradient descent the model in response to the output of the loss function and assist in minimizing the loss function.

⦁	From tensoflow.keras.preprocessing.image import img_to_array 

 Tested the project on 2 different platforms.
 The first is with proteus;
Used a camera with raspberry pi to physically apply the created model. We connected the system to a servo motor so that the camera could rotate 360 ​​degrees.

** Demo videos and source codes are available on file.


The second is to simulate the face recognition program with PyCharm;

When we ran the detecting.py file, the camera turned on and when we put on a mask, the indicator on the screen turned blue by typing 'wearing mask', and when we removed the mask, it gave a 'not wearing mask' warning in red.

** Demo videos and source codes are available on file.




