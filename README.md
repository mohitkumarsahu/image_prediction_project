Description

This file contains a project called image prediction using CNN, which predicts the image given to the model.
It uses Convolutional Neural Network, under which VGG16 pre-trained model was used which contain 1000 classes to different categories like dog, house, lighter etc.
It uses keras - high level deep learning framework used to build and train neural network easily. It is a part of tensorflow.


Code Flow

1. import all the libraries mentioned in requirements.txt.
2. load the model VGG16 and include weights of ImageNet ( this is also a pre-trained model which carry more than 1000 classes, so are just taking all the pre-trained weights directly into VGG16).
3. Now load and prepare the image by converting it into array, expanding dimension, and finally preprocess the image.
4. Predict the image using model.predict(image variable).
5. Decode the generated output using decode_predictions function of keras.
6. Finally show the output.



