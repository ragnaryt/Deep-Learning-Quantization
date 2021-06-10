# Workflow

[1] SavedModel 파일 확보
   1. TFH SM file <https://www.tensorflow.org/lite/performance/post_training_quantization?hl=ko>
   2. 1번 안될시 TF model <https://github.com/tensorflow/models/tree/master/research/slim/nets>

[2] TFL 파일 확보
   1. savedmodel 파일 확보
   2. tflite 파일로 convert  
      i) default -> convert  
      ii) quantization -> convert
   3. 1번 안될시 TFH TFL 파일 사용 <https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224/1/metadata/1>

[3] Inference 실행
   1. tflite 파일 확보
   2. PCA 링크 참조 <https://coral.ai/docs/edgetpu/tflite-python/#overview>
   3. PCA 설치  
      i) sudo apt-get update  
      ii) sudo apt-get install python3-pycoral
   4. PCA 링크의 Inference code 실행
   
# Links

TFL Quantization 	<https://www.tensorflow.org/lite/performance/post_training_quantization?hl=ko>  
PCA 		<https://coral.ai/docs/edgetpu/tflite-python/#overview>  
TFL.I 		<https://www.tensorflow.org/lite/api_docs/python/tf/lite/Interpreter>  
PCA GitHub 	<https://github.com/google-coral/pycoral>  
TF Model 	<https://github.com/tensorflow/models/tree/master/research/slim/nets>  
TFH SM file 	<https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4>  
TFH TFL file 	<https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224/1/metadata/1> 