# Post quantization

## Model: SSD MobileNet V1 FPN 640x640 (pre-trained on the COCO 2017 dataset)
TFLite currently only supports SSD Architectures (excluding EfficientDet) for boxes-based detection
<br></br>

## Environment
### Quantization
Platform: Google Colab  
TensorFlow: 2.3 (to support for TensorFlow Lite Converter)
<br></br>

### Inference
Device: google coral dev board  
OS: Mendel Linux 5.0
<br></br>

## Code
model inspection: [google colab link](https://colab.research.google.com/drive/1jvyqfpibTa_XKEcNtUHJwwRDVLEtufYT?usp=sharing)  
model quantization: [google colab link](https://colab.research.google.com/drive/1QcMs77CbzZyDIPJDFemOSnBLVn1g6JJC?usp=sharing)  
model benchmark: [google colab link](https://colab.research.google.com/drive/1P1KFo15ZAyryICmBjFs3sLHtfhss9nHl?usp=sharing)  
