# Top-level Makefile for medium challenges
# This Makefile builds all subdirectories

# GPU Architecture Selection
# Uncomment ONE of the following lines to target a specific GPU:
# GPU_ARCH = gfx90a   # AMD MI210
# GPU_ARCH = gfx908   # AMD MI100
# GPU_ARCH = gfx1100  # AMD Radeon W7900
# If not set, auto-detection will be used

# Export GPU_ARCH to all subdirectories
ifdef GPU_ARCH
export GPU_ARCH
endif

SUBDIRS = 2d_convolution \
          Batched_Matrix_Multiplication \
          Categorical_Cross_Entropy_Loss \
          gemm \
          histogram \
          Logistic_Regression \
          Matrix_Power \
          Mean_Squared_Error \
          Ordinary_Least_Squares \
          prefix_sum \
          reduction \
          softmax \
          softmax_attention \
		  2D_Max_Pooling \
		  Batch_Normalization

.PHONY: all clean force $(SUBDIRS)

# Default: Stop on first error
all: $(SUBDIRS)

$(SUBDIRS):
	@echo "========== Building $@ =========="
	$(MAKE) -C $@
	@echo ""

# Alternative: Continue on error (use: make force)
force:
	@for dir in $(SUBDIRS); do \
		echo "========== Building $$dir =========="; \
		$(MAKE) -C $$dir || echo "*** Error in $$dir (continuing...)"; \
		echo ""; \
	done

clean:
	@echo "========== Cleaning all subdirectories =========="
	@for dir in $(SUBDIRS); do \
		echo "Cleaning $$dir..."; \
		$(MAKE) -C $$dir clean || true; \
	done
	@echo "Done cleaning."

# Build specific subdirectory
# Usage: make <subdirectory_name>
# Example: make gemm

# Alternative: Set GPU architecture from command line
# Usage: make GPU_ARCH=gfx90a
# Example: make GPU_ARCH=gfx1100
