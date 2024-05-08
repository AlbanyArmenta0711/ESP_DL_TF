#Model:
# input: 784-dimensional vector -> Dense (64 units, ReLu) -> Dense (64 units, ReLu) -> Dense (10 units, softmax) -> class (10 possible)

import numpy as np
import keras 
#from keras import layers
#from keras import ops 
import matplotlib.pyplot  as plt
import numpy as np

#Create an input node using TF Functional API
#inputs = keras.Input(shape=(784,))

#Create the first layer of the net 
#dense = layers.Dense(64, activation="relu")
#x = dense(inputs) #Like drawing the arrow from inputs to the first dense layer, x is the output. 
#Adding the following dense layers
#x = layers.Dense(64, activation = "relu")(x)
#outputs = layers.Dense(10, activation= 'softmax')(x)

#Once layers are defined and connected, creat the model
#model = keras.Model(inputs = inputs, outputs = outputs, name = "mnist_model")
#print the summary of the model
#model.summary()

#Load MNIST image data set
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#ORI_X_valid = x_test
#ORI_y_valid = y_test 
#reshape data loaded into vectors
#x_train = x_train.reshape(60000, 784).astype('float32') / 255
#x_test = x_test.reshape(10000, 784).astype('float32') / 255

#n_classes = 10
#y_train = keras.utils.to_categorical(y_train, n_classes)
#y_test = keras.utils.to_categorical(y_test, n_classes)

x_test = x_test.reshape(10000, 784)
x_const = x_test[300]
y_const = y_test[300]
#print(y_const)

sourceFile = open(str(y_const),'a')
for x in x_const:
    print(x, ",",file=sourceFile, end='')
sourceFile.close()
#specify training configuration
#model.compile(
#    loss= keras.losses.CategoricalCrossentropy(),
#    optimizer = keras.optimizers.RMSprop(), 
#    metrics=["accuracy"],
#)
#train the model by calling fit() function, which will train the model by slicing data into
#batches of size batch_size, and iterating over the dataset for a given number of epochs.
#history object holds a record of the loss values and metric values during training
#history = model.fit(x_train, y_train, batch_size= 64, epochs = 10, validation_split= 0.2)

#idx = 6
#plt.imshow(x_const, cmap='Greys')
#plt.show()
#prediction = model.predict(x_test[idx : (idx + 1)])[0]
#ans = np.argmax(prediction)
#print("ANS:", ans)

#Test the model with x_test
#test_scores = model.evaluate(x_test, y_test,verbose = 2)
#print("Test loss:", test_scores[0])
#print("Test accuracy:", test_scores[1])

#Save the model
#model.export("mnist_model", "tf_saved_model")