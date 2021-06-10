#!/bin/bash
echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: 32bit float"
for ((i=0;i<5;i++))
do
	python3 benchmark.py --model models/32_float_model.tflite  --input images/grace_hopper.jpg --output images/grace_hopper_processed_32_bit.jpg --device edge
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: dynamic range (representative dataset: None)"
for ((i=0;i<5;i++))
do
	python3 benchmark.py --model models/dynamic_model.tflite  --input images/grace_hopper.jpg --output images/grace_hopper_processed_dynamic.jpg --device cpu
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: dynamic range (representative dataset: Random)"
for ((i=0;i<5;i++))
do
	python3 benchmark.py --model models/dynamic_model_rep_random.tflite  --input images/grace_hopper.jpg --output images/grace_hopper_processed_dynamic_random.jpg --device edge
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: dynamic range (representative dataset: COCO)"
for ((i=0;i<5;i++))
do
	python3 benchmark.py --model models/dynamic_model_rep_coco.tflite  --input images/grace_hopper.jpg --output images/grace_hopper_processed_dynamic_coco.jpg --device edge
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: 16bit float (representative dataset: Random)"
for ((i=0;i<5;i++))
do
	python3 benchmark.py --model models/16_float_model_rep_random.tflite  --input images/grace_hopper.jpg --output images/grace_hopper_processed_16_bit_random.jpg --device cpu
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: 16bit float (representative dataset: COCO)"
for ((i=0;i<5;i++))
do
	python3 benchmark.py --model models/16_float_model_rep_coco.tflite --input images/grace_hopper.jpg --output images/grace_hopper_processed_16_bit_coco.jpg --device cpu
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: 8bit int (representative dataset: Random)"
for ((i=0;i<5;i++))
do
	python3 benchmark.py --model models/8_int_model_rep_random.tflite --input images/grace_hopper.jpg --output images/grace_hopper_processed_8_bit_random.jpg --device edge
done

echo ""
echo "**********************************************************************************************"
echo ""

echo "ssd mobilenet v1 coco post quantization: 8bit int (representative dataset: COCO)"
for ((i=0;i<5;i++))
do
	python3 benchmark.py --model models/8_int_model_rep_coco.tflite --input images/grace_hopper.jpg --output images/grace_hopper_processed_8_bit_coco.jpg --device edge
done

