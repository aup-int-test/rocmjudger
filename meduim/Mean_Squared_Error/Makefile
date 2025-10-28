HIPCC = hipcc
CXX = g++
CXXFLAGS = -O2 -Wno-unused-result
HIPFLAGS = -O2 -Wno-unused-result

# GPU Architecture (can be set from parent Makefile or command line)
ifdef GPU_ARCH
HIPFLAGS += --offload-arch=$(GPU_ARCH)
endif

HIP_SOURCES = \
    main.hip \
    fs_main.hip \
	fs_optimized_ws.hip

CPP_SOURCES = \
    fs_serial.cpp

HIP_TARGETS = $(patsubst %.hip,exe_%,$(HIP_SOURCES))
CPP_TARGETS = $(patsubst %.cpp,exe_%,$(CPP_SOURCES))

TARGETS = $(HIP_TARGETS) $(CPP_TARGETS)

all: $(TARGETS)

exe_%: %.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

exe_%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f exe_*
	rm -rf testcases

.PHONY: all clean