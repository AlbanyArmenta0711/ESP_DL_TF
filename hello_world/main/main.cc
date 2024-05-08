#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "esp_log.h"

#include "model.h"

//NUMBER 7!!!
 float number_to_infer[] = {
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,84 ,185 ,159 ,151 ,60 ,36 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,222 ,254 ,254 ,254 ,254 ,241 ,198 ,198 ,198 ,198 ,198 ,198 ,198 ,198 ,170 ,52 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,67 ,114 ,72 ,114 ,163 ,227 ,254 ,225 ,254 ,254 ,254 ,250 ,229 ,254 ,254 ,140 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,17 ,66 ,14 ,67 ,67 ,67 ,59 ,21 ,236 ,254 ,106 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,83 ,253 ,209 ,18 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,22 ,233 ,255 ,83 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,129 ,254 ,238 ,44 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,59 ,249 ,254 ,62 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,133 ,254 ,187 ,5 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,9 ,205 ,248 ,58 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,126 ,254 ,182 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,75 ,251 ,240 ,57 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,19 ,221 ,254 ,166 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,3 ,203 ,254 ,219 ,35 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,38 ,254 ,254 ,77 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,31 ,224 ,254 ,115 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,133 ,254 ,254 ,52 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,61 ,242 ,254 ,254 ,52 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,121 ,254 ,254 ,219 ,40 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,121 ,254 ,207 ,18 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0
 };

//Global variables
namespace {
  const tflite::Model *model = nullptr; 
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int inference_count = 0;
  const char * TAG = "TF model";
  constexpr int kTensorArenaSize = 81 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  const int input_size = 784; 
  int8_t quant_input[input_size];
}

void setup_model(); 

extern "C" void app_main(void) {
  setup_model();
  //Convert input to float values between 0 to 1 
  for (int i = 0; i < input_size; i++){
    number_to_infer[i] =  number_to_infer[i] / 255;
  }


  ESP_LOGI(TAG, "scale = %f, zero_point = %d", input->params.scale, (int) input->params.zero_point); 
  //Quantize test data
  for (int i = 0; i < input_size; i++) {
    //Formula to quantize output
    quant_input[i] = number_to_infer[i] / input->params.scale + input->params.zero_point;
  }
  //Place quantized input into the model's input tensor
  memcpy(input->data.int8, quant_input, input_size);

  //Run inference
  TfLiteStatus invoke_status = interpreter->Invoke(); 
  if (invoke_status != kTfLiteOk) {
    ESP_LOGE(TAG, "Invoke failed");
    return;
  }

  //Process the inference results
  float score_f[10] = {0};
  
  for (int i = 0; i < 10; i++){
    score_f[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    
    ESP_LOGI(TAG, "probability of class %d: %f", i, score_f[i]);
  }



}

void setup_model() {
  //Map the model into a usable data structure
  model = tflite::GetModel(g_model); 
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return; 
  } else {
    ESP_LOGI(TAG, "Model mapped into data structure");
  }
  //Add operations according to the trained net
  static tflite::MicroMutableOpResolver<2> resolver; 
  resolver.AddFullyConnected(); 
  resolver.AddSoftmax();
  //Build the interpreter 
  static tflite::MicroInterpreter static_interpreter (model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter; 
  //Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors(); 
  if (allocate_status != kTfLiteOk) {
    ESP_LOGE(TAG, "Error allocating model tensors");
    return; 
  } else {
    ESP_LOGI(TAG, "Tensors allocated for model"); 
  }

  //Obtain pointers to the model's input and output tensors
  input = interpreter->input(0); 
  output = interpreter->output(0); 
  inference_count = 0; 
}