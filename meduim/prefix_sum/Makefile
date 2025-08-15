HIPCC = hipcc
CXX = g++
CXXFLAGS = -O2
HIPFLAGS = -O2

CUDA_SOURCES = \
    main.cu \
    fs_main.cu \
	fs_optimized_upsweet.cu

CPP_SOURCES = \
    fs_serial.cpp

CUDA_TARGETS = $(patsubst %.cu,exe_%,$(CUDA_SOURCES))
CPP_TARGETS = $(patsubst %.cpp,exe_%,$(CPP_SOURCES))

TARGETS = $(CUDA_TARGETS) $(CPP_TARGETS)

all: $(TARGETS)

exe_%: %.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

exe_%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f exe_*

.PHONY: all clean