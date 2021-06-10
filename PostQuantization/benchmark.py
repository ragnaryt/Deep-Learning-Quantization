import tflite_runtime.interpreter as tflite
from PIL import Image, ImageDraw
import numpy as np
import time
import argparse
import platform

def inference(model, input, output, device):
  if device == 'edge':
    interpreter = tflite.Interpreter(
        model_path=model,
        experimental_delegates=[
            tflite.load_delegate('libedgetpu.so.1')
        ])
  else:
    interpreter = tflite.Interpreter(
        model_path=model
    )

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  interpreter.allocate_tensors()

  def draw_rect(image, box):
      y_min = int(max(1, (box[0] * height)))
      x_min = int(max(1, (box[1] * width)))
      y_max = int(min(height, (box[2] * height)))
      x_max = int(min(width, (box[3] * width)))
      
      # draw a rectangle on the image
      draw = ImageDraw.Draw(image)
      draw.rectangle([(x_min, y_min), (x_max, y_max)], outline='red')

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  img = Image.open(input)
  w, h = img.size
  scale = min(width / w, height / h)
  w, h = int(w * scale), int(h * scale)

  tensor = interpreter.tensor(input_details[0]['index'])()[0]
  tensor.fill(0)
  _, _, channel = tensor.shape
  tensor[:h, :w] = np.reshape(img.resize((w, h)), (h, w, channel))
  
  if floating_model:
      tensor = tensor.astype(np.float32)
      tensor = tensor / 255.0
      tensor = (np.float32(tensor) - 127.5) / 127.5
  
  start_time = time.perf_counter()
  interpreter.invoke()
  stop_time = time.perf_counter()

  rects = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
  rects = np.squeeze(rects)

  scores = interpreter.tensor(interpreter.get_output_details()[2]['index'])()
  scores = np.squeeze(scores)

  img = img.convert('RGB')

  for index, score in enumerate(scores[0]):
    if score > 0.5:
      draw_rect(img,rects[0][index])

  img.save(output)
  print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-d', '--device', required=True)
    args = parser.parse_args()
    inference(args.model, args.input, args.output, args.device)
  
