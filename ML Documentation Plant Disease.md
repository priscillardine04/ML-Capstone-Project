# Plant Disease Detection

## Project Description
Plant disease detection feature built in for Capstone Project. This feature uses a machine learning algorithm by building a Convolutional Neural Network model in a Tensorflow lite format for further deployment to Android Studio. This project was created by Priscilla Ardine Puspitasari, Muhammad Hafizh Rachman, and Qanita Zafa Ariska as part of the Bangkit Capstone Project, demonstrating their skills and knowledge gained throughout the program.

## Features
Plant detection and classification based on uploaded images.

## Splitting Data
With a total of 11.097 images of leaf, we split the data into 3 set, that are Training set with a proportion of 80%, Validation set with a proportion of 10%, and Test set with a proportion of 10%.
``` python
train_dir = 'D:/Hafizh/Dataset/training'
val_dir = 'D:/Hafizh/Dataset/validation'
test_dir = 'D:/Hafizh/Dataset/testing'
image_width, image_height = 150, 150
batch_size = 128
num_epochs = 20
num_classes = len(os.listdir(train_dir))
print(num_classes)
```

## Input Data
Data in the form of images with 45 classes based on plant species and diseases

## Data Augmentation
Preprocessing done by using ImageDataGenerator that were provided by Keras. Here, we doing some data augmentation. For training data, we doing some parameter to the dataset, meanwhile for the validation data and testing data we only doing rescale to the dataset.
``` python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)
```

## Build Model
We are doing transfer learning with MobileNet architecture with several additional layers and dense output with 45 classes and softmax activation. MobileNets is a Convolutional Neural Network (CNN) architecture that focus on optimizing latency but at the same time also yield small networks. 

``` python
IMG_SHAPE=(image_width,image_height, 3)
mobnet_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')
mobnet_model.trainable = True
mobnet_model.summary() 
#Adding Layers
model = Sequential()
model.add(mobnet_model)
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
ckpt = ModelCheckpoint("LeaveDisease.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
history = model.fit(
    train_data,
    steps_per_epoch = train_data.n // batch_size,
    validation_data = val_data,
    validation_steps = val_data.n // batch_size,
    callbacks = [ckpt],
    epochs=num_epochs)

```
![alt text](https://github.com/priscillardine04/ML-Capstone-Project/blob/main/Output%20Model/acc.jpeg?raw=true)

## Model Evaluation
To know the plot of accuracy and loss on training and validation data. The accuracy plot results are good because the graph shows an increase, 
while the loss plot is also good because the graph shows a decrease

``` python
#Accuracy Plot
plt.figure(figsize=(10, 4))
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.legend()
plt.title('Training and Validation Accuracy')
plt.ylabel("Accuracy (Training and Validation)")
plt.xlabel("Epochs")
plt.show()

#Loss Plot
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.legend()
plt.title('Training and Validation Loss')
plt.ylabel("Loss (Training and Validation)")
plt.xlabel("Epochs")
plt.show()
```
![alt text](https://github.com/priscillardine04/ML-Capstone-Project/blob/main/Output%20Model/plot.PNG?raw=true)


## Evaluate Test Data
Evaluate testing data based on accuracy and loss values ​​and the results are quite good.

``` python
loss, accuracy = model.evaluate(test_data)
print('Test Accuracy:', accuracy)
print('Test Loss:', loss)
```
![alt text](https://github.com/priscillardine04/ML-Capstone-Project/blob/main/Output%20Model/test%20acc.jpeg?raw=true)

## Prediction of Test Set
Knowing the prediction results of plant species and diseases from testing data based on the model that has been built

``` python
#set some number first
n = 200

plt.imshow(x_test[n])
plt.show()

true_label = np.argmax(y_test,axis=1)[n]
print("The True Class:",true_label,":",class_names[true_label])

prediction = model.predict(x_test[n][np.newaxis,...])[0]

predicted_label = np.argmax(prediction)
print("The Predicted Class:",predicted_label,":",class_names[predicted_label])

if true_label == predicted_label:
    print("The model can predict the right class")
else:
    print("The model fail to predict the right class")
```
![alt text](https://github.com/priscillardine04/ML-Capstone-Project/blob/main/Output%20Model/predict.jpeg?raw=true)

## Convert Model to Tflite Format
Models that have good accuracy can be used to classify healthy and diseased plant species based on plant leaves. After the prediction value is already good, then 
deployment will be carried out on Android by converting the file to the Tensorflow lite format. The Tensorflow lite format contains defined models in h5 file format

## Contact
For any inquiries or further information, please contact the project developers:

- Priscilla Ardine Puspitasari: [Email](mailto:priscillaardine9784@gmail.com)
- Muhammad Hafizh Rachman: [Email](m.hafizh272@gmail.com)
- Qanita Zafa Ariska: [Email](qanitazafa@gmail.com)

Model in Tflite : [Tensorflow Lite File](https://drive.google.com/drive/folders/1-laQf4w3eVFWU9j8G4XJfQXZzqyPJUna?usp=sharing)
