# Regenerate NVIDIA codec golden tables
#
# Requires CUDA headers with FP4/FP8 types (cuda_fp4.h / cuda_fp8.h):
#
#   g++ -O2 -std=c++17 -I/usr/local/cuda/include \
#       generate_nvidia_codec_goldens.cpp -o generate_nvidia_codec_goldens
#   ./generate_nvidia_codec_goldens > ../test/nvidia_codec_goldens.py
