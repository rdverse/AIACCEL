# compiler
NVCC = nvcc

# compiler flgs
CFLAGS = -arch=sm_60 -std=c++11 -I../utils $(shell pkg-config --cflags opencv4) -diag-suppress=611

LDFLAGS = $(shell pkg-config --libs opencv4)
# Directories
SRCDIR = .
UTILDIR = ../utils
BUILDDIR = build


# source files
SRC = $(SRCDIR)/4_matmul.cu $(UTILDIR)/data_gen.cu

# object files
OBJ = $(BUILDDIR)/4_matmul.o $(BUILDDIR)/data_gen.o

TARGET = 4_matmul

all: $(TARGET)

# Build target
$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

#main dir: Compile cuda source files into object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

# utils dir: Compile CUDA files into object files
$(BUILDDIR)/%.o: $(UTILDIR)/%.cu
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

# # utils dir: Compile C++ files into object files
# $(BUILDDIR)/%.o: $(UTILDIR)/%.cc
# 	mkdir -p $(BUILDDIR)
# 	$(NVCC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rvf $(BUILDDIR) $(TARGET)