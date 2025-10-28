HIPCC = hipcc
HIPFLAGS = -O2 -Wno-unused-result

# GPU Architecture (can be set from parent Makefile or command line)
ifdef GPU_ARCH
HIPFLAGS += --offload-arch=$(GPU_ARCH)
endif

TARGETS = main fs_main fs_optimized_shared fs_optimized_doublebuff

all: $(TARGETS)

main: main.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

fs_main: fs_main.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

fs_optimized_shared: fs_optimized_shared.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

fs_optimized_doublebuff: fs_optimized_doublebuff.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
	rm -rf testcases

.PHONY: all clean
