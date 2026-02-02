CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS := -I../helper_lib
LDFLAGS  := ../helper_lib/helper_lib.a -lm
MAKECMD  = make

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS += -lOpenCL
endif

all: solution

debug: CFLAGS += -DOCL_DEVICE_TYPE=CL_DEVICE_TYPE_CPU
debug: MAKECMD += debug
debug: solution

solution: ../helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../helper_lib/helper_lib.a: 
	cd ../helper_lib; $(MAKECMD)

naive: solution
	./solution Dataset/0/input0.raw Dataset/0/input1.raw Dataset/0/output.raw output.raw naive
	./solution Dataset/1/input0.raw Dataset/1/input1.raw Dataset/1/output.raw output.raw naive
	./solution Dataset/2/input0.raw Dataset/2/input1.raw Dataset/2/output.raw output.raw naive
	./solution Dataset/3/input0.raw Dataset/3/input1.raw Dataset/3/output.raw output.raw naive
	./solution Dataset/4/input0.raw Dataset/4/input1.raw Dataset/4/output.raw output.raw naive
	./solution Dataset/5/input0.raw Dataset/5/input1.raw Dataset/5/output.raw output.raw naive
	./solution Dataset/6/input0.raw Dataset/6/input1.raw Dataset/6/output.raw output.raw naive
	./solution Dataset/7/input0.raw Dataset/7/input1.raw Dataset/7/output.raw output.raw naive
	./solution Dataset/8/input0.raw Dataset/8/input1.raw Dataset/8/output.raw output.raw naive
	./solution Dataset/9/input0.raw Dataset/9/input1.raw Dataset/9/output.raw output.raw naive
	./solution Dataset/10/input0.raw Dataset/10/input1.raw Dataset/10/output.raw output.raw naive

coarsened: solution
	./solution Dataset/0/input0.raw Dataset/0/input1.raw Dataset/0/output.raw output.raw coarsened
	./solution Dataset/1/input0.raw Dataset/1/input1.raw Dataset/1/output.raw output.raw coarsened
	./solution Dataset/2/input0.raw Dataset/2/input1.raw Dataset/2/output.raw output.raw coarsened
	./solution Dataset/3/input0.raw Dataset/3/input1.raw Dataset/3/output.raw output.raw coarsened
	./solution Dataset/4/input0.raw Dataset/4/input1.raw Dataset/4/output.raw output.raw coarsened
	./solution Dataset/5/input0.raw Dataset/5/input1.raw Dataset/5/output.raw output.raw coarsened
	./solution Dataset/6/input0.raw Dataset/6/input1.raw Dataset/6/output.raw output.raw coarsened
	./solution Dataset/7/input0.raw Dataset/7/input1.raw Dataset/7/output.raw output.raw coarsened
	./solution Dataset/8/input0.raw Dataset/8/input1.raw Dataset/8/output.raw output.raw coarsened
	./solution Dataset/9/input0.raw Dataset/9/input1.raw Dataset/9/output.raw output.raw coarsened
	./solution Dataset/10/input0.raw Dataset/10/input1.raw Dataset/10/output.raw output.raw coarsened

optional: solution
	./solution Dataset/0/input0.raw Dataset/0/input1.raw Dataset/0/output.raw output.raw optional
	./solution Dataset/1/input0.raw Dataset/1/input1.raw Dataset/1/output.raw output.raw optional
	./solution Dataset/2/input0.raw Dataset/2/input1.raw Dataset/2/output.raw output.raw optional
	./solution Dataset/3/input0.raw Dataset/3/input1.raw Dataset/3/output.raw output.raw optional
	./solution Dataset/4/input0.raw Dataset/4/input1.raw Dataset/4/output.raw output.raw optional
	./solution Dataset/5/input0.raw Dataset/5/input1.raw Dataset/5/output.raw output.raw optional
	./solution Dataset/6/input0.raw Dataset/6/input1.raw Dataset/6/output.raw output.raw optional
	./solution Dataset/7/input0.raw Dataset/7/input1.raw Dataset/7/output.raw output.raw optional
	./solution Dataset/8/input0.raw Dataset/8/input1.raw Dataset/8/output.raw output.raw optional
	./solution Dataset/9/input0.raw Dataset/9/input1.raw Dataset/9/output.raw output.raw optional
	./solution Dataset/10/input0.raw Dataset/10/input1.raw Dataset/10/output.raw output.raw optional

clean: 
	rm -f solution
	rm -f output.raw
	$(MAKE) -C ../helper_lib clean

time-naive: solution
	python3 ../utils/profile.py  --args ./solution Dataset/10/input0.raw Dataset/10/input1.raw Dataset/10/output.raw output.raw naive

time-coarsened: solution
	python3 ../utils/profile.py  --args ./solution Dataset/10/input0.raw Dataset/10/input1.raw Dataset/10/output.raw output.raw coarsened

time-optional: solution
	python3 ../utils/profile.py  --args ./solution Dataset/10/input0.raw Dataset/10/input1.raw Dataset/10/output.raw output.raw optional
