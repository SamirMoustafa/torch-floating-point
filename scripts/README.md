# Regenerate NVIDIA codec golden tables
#
# Requires CUDA headers with FP4/FP8 types (cuda_fp4.h / cuda_fp8.h):
#
#   g++ -O2 -std=c++17 -I/usr/local/cuda/include \
#       generate_nvidia_codec_goldens.cpp -o generate_nvidia_codec_goldens
#   ./generate_nvidia_codec_goldens > ../test/nvidia_codec_goldens.py

# Explorer JS kernel vs Python/C++ (docs/javascripts/fp-codec.js)
#
# Requires the package (CPU torch is enough) and Node:
#
#   python scripts/gen_fp_codec_goldens.py --write
#   python scripts/check_fp_codec.py
#
# --write is manual after an intentional kernel change. CI always --check.
# When adding a scale_encode or preset, see docs/explorer.md "Maintaining the explorer".
