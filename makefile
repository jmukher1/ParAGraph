# =============================================================================
# Makefile for ABM citation-network simulator (CUDA / Hopper sm_90)
# =============================================================================

# Compiler
NVCC = nvcc #-g -G

# Per-file compile flags. -dlto added here so device LTO IR is actually
# generated at compile time -- it must match the -dlto on the link step below,
# or device link-time optimization silently has nothing to link against.
NVCCFLAGS = -rdc=true -dc -dlto

CUDAFLAGS = -std=c++20 -arch=sm_90 -O3 -use_fast_math --expt-extended-lambda \
            -Xcompiler -fopenmp -lineinfo -Xptxas -v

LIBS     = -lcudart
EXPERIMENTAL = --expt-relaxed-constexpr
LIBDIRS  = -L$(CUDA_HOME)/lib64
LPFLAGS  = -lstdc++

# Final link flags. -dlink removed: this Makefile links objects directly into
# the final executable in one nvcc invocation (main.o's real main() is in
# $(OBJS)), so -dlink ("device-link only, emit an object, no executable") is
# the wrong mode here -- nvcc already auto-detects the relocatable device
# code in the objects and runs nvlink internally without it. -dlto added to
# match NVCCFLAGS above.
NVLINKFLAGS = --expt-extended-lambda -lcudadevrt -lcudart \
              -Xcompiler -fopenmp -lgomp -arch=sm_90 -dlto

SRC_DIR   = src
BUILD_DIR = build
INC_DIR   = ./includes

# Single INCDIRS definition (the earlier duplicate -- referencing an
# undefined $(CCCL_HOME) -- has been removed; it was silently overwritten
# by this one anyway since Make is last-assignment-wins).
INCDIRS = -I$(INC_DIR) -I$(SRC_DIR) -I$(CUCO_HOME)/include \
          -I$(CUCO_HOME)/build/_deps/cccl-src/libcudacxx \
          -I$(CUCO_HOME)/build/_deps/cccl-src/libcudacxx/include \
          -I$(CUCO_HOME)/build/_deps/cccl-src/thrust \
          -I$(CUCO_HOME)/build/_deps/cccl-src/cub

TARGET = abm

# Files
LINK_OBJS = $(BUILD_DIR)/utils.o $(BUILD_DIR)/graph.o $(BUILD_DIR)/abm.o \
            $(BUILD_DIR)/kernel.o $(BUILD_DIR)/main.o
OBJS = $(LINK_OBJS)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CUDAFLAGS) $^ $(LIBDIRS) $(INCDIRS) $(LIBS) -o $@ $(NVLINKFLAGS) $(EXPERIMENTAL)

# Ensure build/ exists before any object rule tries to write into it
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.cu $(INC_DIR)/utils.cuh | $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) $(NVCCFLAGS) -c $(INCDIRS) $< -o $@ $(EXPERIMENTAL)

$(BUILD_DIR)/graph.o: $(SRC_DIR)/graph.cu $(INC_DIR)/graph.cuh $(INC_DIR)/node.cuh | $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) $(NVCCFLAGS) -c $(INCDIRS) $< -o $@ $(EXPERIMENTAL)

$(BUILD_DIR)/abm.o: $(SRC_DIR)/abm.cu $(INC_DIR)/abm.cuh | $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) $(NVCCFLAGS) -c $(INCDIRS) $< -o $@ $(EXPERIMENTAL)

$(BUILD_DIR)/kernel.o: $(SRC_DIR)/kernel.cu $(INC_DIR)/utils.cuh $(INC_DIR)/int2.cuh $(INC_DIR)/abm.cuh | $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) $(NVCCFLAGS) -c $(INCDIRS) $< -o $@ $(EXPERIMENTAL)

# main.o now gets $(EXPERIMENTAL) too, for consistency with every other
# object rule (it happened not to need --expt-relaxed-constexpr as of the
# last successful build log, but there's no reason to special-case it and
# it'll bite you silently if main.cu ever needs it after a header change).
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu $(INC_DIR)/abm.cuh $(INC_DIR)/argparse.h | $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) $(NVCCFLAGS) -c $(INCDIRS) $< -o $@ $(EXPERIMENTAL)

clean:
	rm -rf $(TARGET) $(BUILD_DIR)/*.o