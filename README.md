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
El código que se discute en toda esta sección se presenta en el archivo **hello_world_tf.py**. Para comenzar con el tutorial basta con ejecutar primero dicho script. 

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

La siguiente figura es una representación de la red generada a partir del las líneas de código presentadas:

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

## Conversión con TensorFlow Lite
El código que se discute en esta sección se encuentra en el archivo **lite_converter.py**. Para hacer su conversión, se puede ejecutar directamente este script después de haber generado el modelo con el script **hello_word_tf.py**. 

Para realizar la conversión el primer paso es cargar el modelo generado por el script anterior:

    MODEL_PATH = "mnist_model"
    #Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH)

Después se debe configurar los parámetros para la cuantización, esto es, convertir todos los valores flotantes que componen los pesos de la red a enteros de 8 bits, así como definir que las entradas y salidas de la red de igual manera serán enteros de 8 bits. Este paso se recomienda ya que los recursos con los que pudiera contar el dispositivo, memoria y CPU, pueden ser limitados, por lo que cuantizar el modelo lo hace más ligero y rápido de ejecutarse. 

    def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100):
        yield [input_value]

    #Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH)
    #Model Quantization for integer only
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8 
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    #Save the model
    with open('model_quant.tflite', 'wb') as f:
        f.write(tflite_quant_model)

El modelo cuantizado se habrá guardado con el nombre *model_quant*. Sin embargo, aún no se encuentra listo para ser cargado al ESP32. Para ello, es necesario convertir el modelo mediante un tercer script encontrado en [stackoverflow](https://stackoverflow.com/questions/73301347/how-to-convert-model-tflite-to-model-cc-and-model-h-on-windows-10) que da como resultado el archivo .cc con el modelo convertido a vector a partir del archivo binario generado. Este script se encuentra en el repositorio bajo el nombre de **quant_model.py**. 

**Archivo model.h**

    extern const unsigned char g_model[];
    extern const int g_model_len;

**Archivo model.cc**
    
    #include "model.h"

    const unsigned char g_model[] = {
        0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
        ...
    };

    const int g_model_len = 57960; 

## Transferencia al ESP32 
Para desplegar el modelo, se tomo como base el ejemplo *hello_world* de [esp-tflite-micro](https://github.com/espressif/esp-tflite-micro/tree/master), el cual contiene las dependencias necesarias para ejecutar modelos de TensorFlow Lite. Se recomienda que el ejemplo *hello_world* se utilice como base para proyectos que requieran ejecutar modelos de TensorFlow Lite. Todos los archivos asociados al proyecto elaborado con la herramienta ESP-IDF se encuentran bajo el directorio **hello_world**, mientras que el código principal se encuentra en el archivo **main.cc**.

Es necesario contruir el proyecto utilizando la ESP-IDF toolchain para posteriormente cargar el programa al ESP32 y comenzar a monitorear el dispositivo por medio de esta misma toolchain. La siguiente imagen muestra el resultado de cargar y ejecutar el programa en el dispositivo. 

![Resultados de inferencia en el dispositivo](/Images/model_inference.png)
En esta se observa la probabilidad que una imagen proporcionada mediante un vector en el archivo **main.cc** pertenezca a cada una de las posibles clases. Para este caso, se ingreso un vector representativo para un número 7 y el modelo infiere con un 99.60% de certeza que el número ingresado se trata de un 7.
