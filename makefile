# Makefile for Citation ABM with MASS CUDA

# MASS Library Configuration
MASS_DIR = ./../../mass_cuda_core
MASS_INCLUDE = $(MASS_DIR)/src
MASS_LIB = $(MASS_DIR)/lib/mass

# ── Boost Configuration ───────────────────────────────────────────────────
# IMPORTANT: as of this Makefile, only ONE Boost install on this cluster
# actually ships compiled libboost_log*/libboost_thread*/libboost_system*
# libraries -- the 1.85.0 installs are headers-only (that's why
# mass_cuda_core builds fine against them: it only compiles to a static
# .a archive and never actually LINKS against Boost's shared libs).
# citation_abm links a real executable, so it needs the one Boost install
# that has real .so/.a files:
#   /apps/spack/gautschi-cpu/apps/boost/1.86.0-gcc-11.4.1-kpkvtzz
# Confirmed via:
#   find /apps/spack/gautschi-cpu/apps/boost/ -maxdepth 4 -iname "libboost_log*"
# If this path changes (e.g. after a `module load boost` update, or if a
# newer Boost install gains compiled libs), update BOOST_DIR below --
# don't rely on whatever `module load boost` puts on the default search
# path, since that may silently point at a headers-only install again.
#
# This Boost install was built with gcc 11.4.1, so build citation_abm with
# `module load gcc/11` loaded (matches what's currently active based on
# your session) to avoid ABI mismatches between headers and the linked .so.
BOOST_DIR = /apps/spack/gautschi-cpu/apps/boost/1.86.0-gcc-11.4.1-kpkvtzz
BOOST_INCLUDE = $(BOOST_DIR)/include
BOOST_LIB = $(BOOST_DIR)/lib

# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA architecture (adjust for your GPU)
CUDA_ARCH = -arch=sm_90 -gencode=arch=compute_60,code=sm_60

# Compilation flags
NVCC_FLAGS = -std=c++20 --extended-lambda --expt-relaxed-constexpr -O3 -DBOOST_LOG_DYN_LINK \
             -Xcompiler -fPIC -I$(MASS_INCLUDE) -I$(BOOST_INCLUDE)

CUDA_LIBS = -lcudart -lcurand

# ── Link flags ────────────────────────────────────────────────────────────
# -L$(BOOST_LIB): tells the linker where to FIND the Boost libs at link time
# -Wl,-rpath,$(BOOST_LIB): embeds that path into the binary so it can find
#   the .so files again at RUN time, without needing LD_LIBRARY_PATH set
#   in every shell/sbatch script that runs the executable
# Boost libs listed ONCE here (previously duplicated across LDFLAGS and
# MASS_LIBS, which was harmless but redundant)
MASS_LIBS = -L$(MASS_LIB) -lmass_cuda
BOOST_LIBS = -L$(BOOST_LIB) -Xlinker -rpath -Xlinker $(BOOST_LIB) \
             -lboost_log -lboost_log_setup -lboost_thread -lboost_system -lpthread

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
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -I$(INC_DIR) -o $@ $(SOURCES) $(CUDA_LIBS) $(MASS_LIBS) $(BOOST_LIBS)
	@echo "Built successfully: $(TARGET)"

# Run simulation
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

# Check Boost installation actually has compiled libs (not just headers)
check-boost:
	@echo "Checking Boost install: $(BOOST_DIR)"
	@test -d "$(BOOST_INCLUDE)" && echo "  headers found" || echo "  ERROR: headers NOT found at $(BOOST_INCLUDE)"
	@ls $(BOOST_LIB)/libboost_log* >/dev/null 2>&1 && echo "  compiled libs found" || echo "  ERROR: no compiled libs at $(BOOST_LIB) -- this Boost install is headers-only, update BOOST_DIR"

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

