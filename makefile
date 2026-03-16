# Compiler settings
CC = g++
OMPFLAGS=-fopenmp
CXXFLAGS = -O3 -std=c++20  
INCDIRS = -I.
LIBS= -lpthread -lcudart -lcublas
LIBDIRS=-L$(CUDA_HOME)/lib64
LPFLAGS = -lstdc++
NVLINKFLAGS= -lcudadevrt -lcudart
SRC_DIR = src
BUILD_DIR = build
INC_DIR = ./includes
INCDIRS=-I$(CUDA_HOME)/include -I$(INC_DIR)


# Default target
TARGET = abm

# Files
OBJS = $(BUILD_DIR)/main.o $(BUILD_DIR)/abm.o $(BUILD_DIR)/graph.o 

$(TARGET): $(OBJS)
	$(CC) $(CXXFLAGS) $(OMPFLAGS) $^ $(LIBDIRS) $(INCDIRS) $(LIBS) -o $@ $(NVLINKFLAGS)

$(BUILD_DIR)/graph.o: $(SRC_DIR)/graph.cpp $(INC_DIR)/graph.h
	$(CC) $(CXXFLAGS) -c $(INCDIRS) $< -o $@

$(BUILD_DIR)/abm.o: $(SRC_DIR)/abm.cpp $(INC_DIR)/abm.h
	$(CC) $(CXXFLAGS) $(OMPFLAGS) -c $(INCDIRS) $< -o $@

# Compile CUDA files
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(INC_DIR)/abm.h $(INC_DIR)/argparse.h
	$(CC) $(CXXFLAGS) $(OMPFLAGS) -c $(INCDIRS) $< -o $@

clean:
	rm -rf $(TARGET) $(BUILD_DIR)/*.o
