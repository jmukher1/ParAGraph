# Makefile for Citation ABM with MASS CUDA

# MASS Library Configuration
MASS_DIR = ./../mass_cpp_core
MASS_INCLUDE = $(MASS_DIR)/source
MASS_LIB = $(MASS_DIR)/ubuntu
SSH_LIB = $(MASS_LIB)/ssh2/lib

# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA architecture (adjust for your GPU)
CUDA_ARCH = -arch=sm_60 -gencode=arch=compute_60,code=sm_60 \
            -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_86,code=sm_86

# Compilation flags
NVCC_FLAGS = -std=c++20 --extended-lambda --expt-relaxed-constexpr -O3 \
             -Xcompiler -fPIC -I$(MASS_INCLUDE)

CUDA_LIBS = -lcudart -lcurand
MASS_LIBS = -L$(MASS_LIB) -L$(SSH_LIB) -lmass -lssh2

# Directories
BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj
INC_DIR = includes
# Targets
TARGET = $(BIN_DIR)/citation_abm_mass
SOURCES = src/citation_abm_mass.cu
HEADERS = includes/Paper.h

# Default target
all: directories $(TARGET)

# Create directories
directories:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)

# Build main executable
$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -I$(INC_DIR) -o $@ $(SOURCES) $(CUDA_LIBS) $(MASS_LIBS)
	@echo "Built successfully: $(TARGET)"

# Run simulation
#run: $(TARGET)
run:
	./$(TARGET) --edgelist sj_edgelist --nodelist sj_nodelist --out-degree-bag tcen_at_least_five --recency-probabilities sj_recprob --same-year-proportion 0.12 --alpha 0.5 --preferential-weight 0.33 --recency-weight 0.33 --fitness-weight 0.33 --growth-rate 0.02 --auxiliary-information-file dummy2-2.aux --num-cycles 2 --output-file res2-2.out --log-file output2-2.log --num-processors 1 --log-level 1


# Run with custom parameters
run-custom: $(TARGET)
	./$(TARGET) 1000 20 0.05

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Clean outputs
clean-output:
	rm -f nodes_output.csv edges_output.csv

# Clean everything
clean-all: clean clean-output

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc || echo "ERROR: nvcc not found"
	@nvcc --version || echo "ERROR: Cannot run nvcc"
	@nvidia-smi || echo "ERROR: Cannot run nvidia-smi"

# Show GPU info
gpu-info:
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Build with debug symbols
debug: NVCC_FLAGS += -g -G -lineinfo
debug: $(TARGET)
	@echo "Built with debug symbols"

# Profile target
profile: $(TARGET)
	nvprof --print-gpu-trace ./$(TARGET)

# Help
help:
	@echo "Citation ABM MASS CUDA Makefile"
	@echo "================================"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build the executable"
	@echo "  make standalone   - Build without MASS library dependency"
	@echo "  make run          - Build and run simulation"
	@echo "  make run-custom   - Run with custom parameters"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-output - Remove output files"
	@echo "  make clean-all    - Remove everything"
	@echo "  make check-cuda   - Verify CUDA installation"
	@echo "  make gpu-info     - Show GPU information"
	@echo "  make debug        - Build with debug symbols"
	@echo "  make profile      - Build and profile with nvprof"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  MASS_DIR: $(MASS_DIR)"
	@echo "  CUDA_ARCH: Volta/Turing/Ampere (sm_70/75/80/86)"
	@echo ""
	@echo "Note: If MASS is not installed, use 'make standalone'"

.PHONY: all directories run run-custom clean clean-output clean-all \
        check-cuda gpu-info debug profile help standalone
