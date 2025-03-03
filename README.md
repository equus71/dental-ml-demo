# Dental demo

# Dental X-ray Analysis Demo

This demo processes panoramic dental X-rays.

Key elements:
+ Mandible segmentation
+ Teeth detection
+ Teeth segmentation
+ Tooth line extraction
+ Matching & aligning two panoramic X-rays

The entrypoint for the demo is at `dental_3d_analysis.py`

# Requirments

Beyond what is in the `requeirments.txt`, one needs to install also PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection/) and LightGlue (https://github.com/cvg/LightGlue).

I was testing this setup with CUDA 12.2.