# Tutorial para ejecutar modelos de TensorFlow en ESP32 con TensorFlow Lite

## Instalación de Requerimentos
A continuación se listan los requerimientos para realizar el entrenamiento de modelos en Python utilizando TensorFlow.
* [Python 3.11.2](https://www.python.org/downloads/release/python-3112/): fue la versión con la que se realizó este tutorial.
* Numpy: se puede instalar desde el gestor de paqueterias pip.
    >pip install numpy
* TensorFlow:
    >pip install tensorflow
* Keras
    >pip install keras

## Entrenando el Primer Modelo
El código que se discute en toda esta sección se presenta en el archivo **hello_world_tf.py**.

Para generar el modelo de aprendizaje en tensorflow para este tutorial se tienen las siguientes líneas de código en el archivo **hello_world_tf.py**:

    #Create an input node using TF Functional API
    inputs = keras.Input(shape=(784,))
    #Create the first layer of the net 
    dense = layers.Dense(64, activation="relu")
    x = dense(inputs) #Like drawing the arrow from inputs to the first dense layer, x is the output. 
    #Adding the following dense layers
    x = layers.Dense(64, activation = "relu")(x)
    outputs = layers.Dense(10, activation= 'softmax')(x)
    #Once layers are defined and connected, create the model
    model = keras.Model(inputs = inputs, outputs = outputs, name = "mnist_model")

Dicho código da como resultado la red que se muestra en la siguiente figura:

![Modelo Generado](/Images/model.png)

La red tiene un nodo de entrada con una dimensión de 784x1. La salida de este nodo de entrada van a una capa completamente conectada con 64 neuronas y una función de activación ReLu. A su vez, esta capa se conecta a otra capa completamente conectada con exactamente el mismo número de neuronas y misma función de activación, y termina con otra capa completamente conectada pero esta vez de 10 unidades, las cuales corresponden a las etiquetas de clase por lo que se utiliza una función Softmax. 

Esta red utiliza el *dataset de mnist* de dígitos del 0 al 9 escritos a mano para su entrenamiento. Asimismo se evalúa con una partición de prueba de este mismo dataset. Cada imagen del dataset tiene una dimensión de 28x28 en escala de grises, por lo que se redimensiona la imagen para representarla como un vector: 

    #Load MNIST image data set
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    #reshape data loaded into vectors
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

Una vez definido el modelo y habiendo redimensionado los datos se especifica la configuración de entrenamiento y se entrena el modelo, definiendo el tamaño del batch y el número de epocas para el entrenamiento:

    #specify training configuration
    model.compile(
        loss= keras.losses.CategoricalCrossentropy(),
        optimizer = keras.optimizers.RMSprop(), 
        metrics=["accuracy"],
    )
    #train the model by calling fit() function, which will train the model by slicing data into
    #batches of size batch_size, and iterating over the dataset for a given number of epochs.
    #history object holds a record of the loss values and metric values during training
    history = model.fit(x_train, y_train, batch_size= 64, epochs = 10, validation_split= 0.2)

Por último, para evaluar el modelo entrenado, se utiliza la partición para pruebas del dataset de mnist y se exporta el modelo para su posterior conversión en un modelo más ligero y cuantizado.

    #Test the model with x_test
    test_scores = model.evaluate(x_test, y_test,verbose = 2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    #Save the model
    model.export("mnist_model", "tf_saved_model")
La última instrucción genera un directorio con toda la información del modelo en la raíz del proyecto.

![Resultados de Entrenamiento y Prueba](/Images/training_test_results.png)



