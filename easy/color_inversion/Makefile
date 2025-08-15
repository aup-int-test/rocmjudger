HIPCC = hipcc
CXXFLAGS = -O2

SOURCES = \
    main.cu \
	fs_main.cu \
	optimize_coalesdmem.cu \
	fs_optimize_coalesdmem.cu

TARGETS = $(patsubst %.cu,exe_%,$(SOURCES))

all: $(TARGETS)

exe_%: %.cu
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f exe_*

.PHONY: all clean