CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS := -I../../helper_lib
LDFLAGS  := ../../helper_lib/helper_lib.a -lm

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS += -lOpenCL
endif

all: raytracer_parallel

raytracer_parallel: ../../helper_lib/helper_lib.a main.c lib/vec_ops.c 
	$(CC) $(CFLAGS) -o raytracer_parallel main.c $(INCFLAGS) $(LDFLAGS)

../../helper_lib/helper_lib.a: 
	cd ../../helper_lib; make

gpu: raytracer_parallel
	./raytracer_parallel gpu

cpu: raytracer_parallel
	./raytracer_parallel cpu

clean:
	cd ../../helper_lib; make clean
	rm -f raytracer_parallel