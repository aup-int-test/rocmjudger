HIPCC = hipcc
HIPFLAGS = -O2 -Wno-unused-result

# GPU Architecture (can be set from parent Makefile or command line)
ifdef GPU_ARCH
HIPFLAGS += --offload-arch=$(GPU_ARCH)
endif

SOURCES = \
    main.cu \
	fs_main.cu \
	optimize_coalesdmem.cu \
	fs_optimize_coalesdmem.cu

TARGETS = $(patsubst %.cu,exe_%,$(SOURCES))

all: $(TARGETS)

exe_%: %.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

clean:
	rm -f exe_*
	rm -rf testcases

.PHONY: all clean