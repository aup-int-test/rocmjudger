HIPCC = hipcc
HIPFLAGS = -O2 -Wno-unused-result

# GPU Architecture (can be set from parent Makefile or command line)
ifdef GPU_ARCH
HIPFLAGS += --offload-arch=$(GPU_ARCH)
endif

SOURCES = \
    main.hip \
    fs_main.hip \
    optimize_sharedmem.hip \
    fs_optimize_sharedmem.hip \
    optimize_warpshuffle.hip

TARGETS = $(patsubst %.hip,exe_%,$(SOURCES))

all: $(TARGETS)

exe_%: %.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

clean:
	rm -f exe_*
	rm -rf testcases

.PHONY: all clean
