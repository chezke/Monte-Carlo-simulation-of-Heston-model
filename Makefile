# Heston Monte Carlo (CUDA) — build from repository root
NVCC       ?= nvcc
NVFLAGS    := -O3 -std=c++14 -Iinclude
# Optional: NVFLAGS += -arch=sm_75
SRCDIR     := src
BUILDDIR   := bin

SOURCES   := $(wildcard $(SRCDIR)/MC_*.cu)
BINARIES  := $(patsubst $(SRCDIR)/MC_%.cu,$(BUILDDIR)/MC_%,$(SOURCES))

.PHONY: all clean

all: $(BINARIES)

$(BUILDDIR)/MC_%: $(SRCDIR)/MC_%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm -rf $(BUILDDIR)
