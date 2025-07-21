HIPCC = hipcc
CXXFLAGS = -O2

SOURCES = \
    main.cu \
	fs_main.cu

TARGETS = $(patsubst %.cu,exe_%,$(SOURCES))

all: $(TARGETS)

exe_%: %.cu
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f exe_*

.PHONY: all clean