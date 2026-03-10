CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS := -I../helper_lib
LDFLAGS  := ../helper_lib/helper_lib.a -lm

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS += -lOpenCL
endif

all: solution

helper_lib:  ../helper_lib/helper_lib.a 
	cd ../helper_lib; make


solution: opencl_functions/opencl-functions.cpp opencl_functions/ocl_wrapper_torch.cpp
	pip config set global.break-system-packages true
	pip install -v . --no-build-isolation


gpt: solution
	python3 inferance_gpt2.py

run: solution
	python3 test.py

clean: 
	rm -rf build
	rm -rf PA8.egg-info
	rm -rf opencl_functions.egg-info
	rm opencl_functions.cpython-312-x86_64-linux-gnu.so
	cd ../helper_lib; make clean

time:
	python3 ../utils/profile.py -n 2 --args python3 test.py
