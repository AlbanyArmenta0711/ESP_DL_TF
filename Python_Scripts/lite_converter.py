import tensorflow as tf
import keras 


MODEL_PATH = "mnist_model"

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.reshape(10000, 784).astype("float32")/ 255

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

