# Deep Learning Quantization
## Board: Google Coral Dev Board  

## TensorFlow models on the Edge TPU
### Overview
In order for the Edge TPU to provide high-speed neural network performance with a low-power cost, the Edge TPU supports a specific set of neural network operations and architectures.  
The Edge TPU is capable of executing deep feed-forward neural networks such as convolutional neural networks (CNN). It supports only TensorFlow Lite models that are fully 8-bit quantized and then compiled specifically for the Edge TPU.  
You cannot train a model directly with TensorFlow Lite; instead you must convert your model from a TensorFlow file (such as a .pb file) to a TensorFlow Lite file (a .tflite file), using the TensorFlow Lite converter.  
<img src=https://coral.ai/static/docs/images/edgetpu/compile-workflow.png>  

### Quantization
**Full integer post-training quantization** doesn't require any modifications to the network, so you can use this technique to convert a previously-trained network into a quantized model. However, this conversion process requires that you supply a representative dataset.  
That is, you need a dataset that's formatted the same as the original training dataset (uses the same data range) and is of a similar style (it does not need to contain all the same classes, though you may use previous training/evaluation data).  
This representative dataset allows the quantization process to measure the dynamic range of activations and inputs, which is critical to finding an accurate 8-bit representation of each weight and activation value.  
However, not all TensorFlow Lite operations are currently implemented with an integer-only specification (they cannot be quantized using post-training quantization).  
By default, the TensorFlow Lite converter leaves those operations in their float format, which is not compatible with the Edge TPU.  
The Edge TPU Compiler stops compiling when it encounters an incompatible operation (such as a non-quantized op), and the remainder of the model executes on the CPU. So to enforce integer-only quantization, you can instruct the converter to throw an error if it encounters a non-quantizable operation.  

### Float input and output tensors
The Edge TPU requires 8-bit quantized input tensors. However, if you pass the Edge TPU Compiler a model that's internally quantized but still uses float inputs, the compiler leaves a quantize op at the beginning of your graph (which runs on the CPU). Likewise, the output is dequantized at the end.  
So it's okay if your TensorFlow Lite model uses float inputs/outputs. However, if you run an inference with the Edge TPU Python API, that API requires all input data be in uint8 format. You can instead use the TensorFlow Lite API, which provides full control of the input tensors, allowing you to pass your model float inputs—the on-CPU quantize op then converts the input to int8 for processing on the Edge TPU.  
But beware that if your model uses float input and output, then there will be some amount of latency added due to the data format conversion, though it should be negligible for most models (the bigger the input tensor, the more latency you'll see). So to achieve the best performance possible, we recommend fully quantizing your model so the input and output use int8 or uint8 data.

### Compiling
At the first point in the model graph where an unsupported operation occurs, the compiler partitions the graph into two parts. The first part of the graph that contains only supported operations is compiled into a custom operation that executes on the Edge TPU, and everything else executes on the CPU.  
*Currently, the Edge TPU compiler cannot partition the model more than once, so as soon as an unsupported operation occurs, that operation and everything after it executes on the CPU, even if supported operations occur later.*  
<img src=https://coral.ai/static/docs/images/edgetpu/compile-tflite-to-edgetpu.png>  
*If part of your model executes on the CPU, you should expect a significantly degraded inference speed compared to a model that executes entirely on the Edge TPU.*  
When compilation completes, the Edge TPU compiler tells you how many operations can execute on the Edge TPU and how many must instead execute on the CPU (if any at all). But beware that the percentage of operations that execute on the Edge TPU versus the CPU does not correspond to the overall performance impact—if even a small fraction of your model executes on the CPU, it can potentially slow the inference speed by an order of magnitude (compared to a version of the model that runs entirely on the Edge TPU).
<br></br>

## TensorFlow Lite converter
The TensorFlow Lite converter takes a TensorFlow model and generates a TensorFlow Lite model (an optimized FlatBuffer format identified by the .tflite file extension)  
<img src=https://www.tensorflow.org/lite/images/convert/convert.png>
<br></br>

## TensorFlow Lite delegate
By default, TensorFlow Lite executes each model on the CPU. In order to use TensorFlow Lite on google coral dev board, you should use **TensorFlow Lite delegate**.
<br></br>

## Load TensorFlow Lite and run an inference
~~~python
interpreter = tflite.Interpreter(model_path,
  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
~~~
The file passed to load_delegate() is the Edge TPU runtime library, and you installed it when you first set up your device. The filename you must use here depends on your host operating system.
<br></br>

## Edge TPU Compiler
The Edge TPU Compiler (edgetpu_compiler) is a command line tool that compiles a TensorFlow Lite model (.tflite file) into a file that's compatible with the Edge TPU.
<br></br>

## Full integer quantization
For full integer quantization, you need to measure the dynamic range of activations and inputs by supplying sample input data to the converter. Refer to the representative_dataset_gen() function used in the following code.  

### Integer with float fallback (using default float input/output)
In order to fully integer quantize a model, but use float operators when they don't have an integer implementation (to ensure conversion occurs smoothly), use the following steps:
~~~python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
~~~  
*This tflite_quant_model won't be compatible with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU) because the input and output still remain float in order to have the same interface as the original float only model.*  

### Integer only
*Creating integer only models is a common use case for TensorFlow Lite for Microcontrollers and Coral Edge TPUs.*  
Additionally, to ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU), you can enforce full integer quantization for all ops including the input and output, by using the following steps:
~~~python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()
~~~

## TensorFlow Lite 8-bit quantization specification  
### Specification summary
8-bit quantization approximates floating point values using the following formula.  
**real_value = (int8_value - zero_point) * scale**


## Reference
https://www.tensorflow.org/lite/guide?hl=ko  
https://coral.ai/docs/
https://github.com/google-coral/tflite/tree/master/python/examples/detection  
https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_classification_ptq_tf2.ipynb  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md
