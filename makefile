# Compiler settings
NVCC = nvcc #-g -G
CXXFLAGS = -O3 -std=c++20 -dc -x cu
NVCCFLAGS= -rdc=true -dc
CUDAFLAGS= -std=c++20 -arch=sm_80 -O3 -use_fast_math --maxrregcount=128 --expt-extended-lambda -Xcompiler -fopenmp -lineinfo
NVCCLINKFLAGS= -arch=sm_80 -dlink
LIBS= -lcudart -lcublas
EXPERIMENTAL= --expt-relaxed-constexpr
LIBDIRS=-L$(CUDA_HOME)/lib64
LPFLAGS = -lstdc++
NVLINKFLAGS= --expt-extended-lambda -lcudadevrt -lcudart -Xcompiler -fopenmp -lgomp
SRC_DIR = src
BUILD_DIR = build
INC_DIR = ./includes
INCDIRS=-I$(INC_DIR) -I$(SRC_DIR) -I$(CUCO_HOME)/include -I $(CCCL_HOME)/libcudacxx -I $(CCCL_HOME)/thrust -I $(CCCL_HOME)
INCDIRS=-I$(INC_DIR) -I$(SRC_DIR) -I$(CUCO_HOME)/include -I $(CUCO_HOME)/build/_deps/cccl-src/libcudacxx -I $(CUCO_HOME)/build/_deps/cccl-src/libcudacxx/include -I $(CUCO_HOME)/build/_deps/cccl-src/thrust -I $(CUCO_HOME)/build/_deps/cccl-src/cub


# Default target
TARGET = abm

# Files
LINK_OBJS = $(BUILD_DIR)/utils.o $(BUILD_DIR)/graph.o $(BUILD_DIR)/abm.o $(BUILD_DIR)/kernel.o  $(BUILD_DIR)/main.o    
#LINKED_OBJ = $(BUILD_DIR)/link.o
OBJS = $(LINK_OBJS) #$(LINKED_OBJ)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_GEN_FLAGS) $(CUDAFLAGS) $^ $(LIBDIRS) $(INCDIRS) $(LIBS) -o $@ $(NVLINKFLAGS) $(EXPERIMENTAL)

$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.cu $(INC_DIR)/utils.cuh
	$(NVCC) $(CUDAFLAGS) $(NVCCFLAGS) -c $(INCDIRS) $< -o $@ $(EXPERIMENTAL)

$(BUILD_DIR)/graph.o: $(SRC_DIR)/graph.cu $(INC_DIR)/graph.cuh $(INC_DIR)/node.cuh  
	$(NVCC) $(CUDAFLAGS) $(NVCCFLAGS) -c $(INCDIRS) $< -o $@ $(EXPERIMENTAL)

$(BUILD_DIR)/abm.o: $(SRC_DIR)/abm.cu $(INC_DIR)/abm.cuh
	$(NVCC) $(CUDAFLAGS) $(NVCCFLAGS) -c $(INCDIRS) $< -o $@ $(EXPERIMENTAL)

$(BUILD_DIR)/kernel.o: $(SRC_DIR)/kernel.cu $(INC_DIR)/utils.cuh $(INC_DIR)/int2.cuh $(INC_DIR)/abm.cuh 
	$(NVCC) $(CUDAFLAGS) $(NVCCFLAGS) -c $(INCDIRS) $< -o $@ $(EXPERIMENTAL)

# Compile CUDA files
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu $(INC_DIR)/abm.cuh $(INC_DIR)/argparse.h
	$(NVCC) $(CUDAFLAGS) $(NVCCFLAGS) -c $(INCDIRS) $< -o $@


clean:
	rm -rf $(TARGET) $(BUILD_DIR)/*.o
