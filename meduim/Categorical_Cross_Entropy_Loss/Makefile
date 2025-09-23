HIPCC = hipcc
CXX = g++
CXXFLAGS = -O2
HIPFLAGS = -O2

HIP_SOURCES = \
    main.hip \
    fs_main.hip

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

.PHONY: all clean