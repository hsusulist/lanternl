# Makefile for LanternL luaTL CUDA backend

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++14 --shared -Xcompiler -fPIC --use_fast_math -lineinfo

# Default to sm_75 (Google Colab T4 / RTX 20 series), fallback to sm_86 (RTX 30), sm_89 (RTX 40)
ARCHS = -gencode arch=compute_75,code=sm_75 \
				-gencode arch=compute_86,code=sm_86 \
				-gencode arch=compute_89,code=sm_89

TARGET = ai/gpu/luaTL.so
SOURCE = cuda/luaTL_train.cu

.PHONY: all gpu clean

all: gpu

gpu:
		$(NVCC) $(NVCC_FLAGS) $(ARCHS) $(SOURCE) -o $(TARGET)
		@echo "Compilation successful: $(TARGET)"

clean:
		rm -f $(TARGET)