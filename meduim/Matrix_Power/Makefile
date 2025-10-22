HIPCC = hipcc
CFLAGS = -O2

TARGETS = main fs_main fs_optimized_shared

all: $(TARGETS)

main: main.hip
	$(HIPCC) $(CFLAGS) -o $@ $<

fs_main: fs_main.hip
	$(HIPCC) $(CFLAGS) -o $@ $<

fs_optimized_shared: fs_optimized_shared.hip
	$(HIPCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
