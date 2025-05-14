# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -arch=sm_60 -std=c++11 -I../utils

# Directories
SRCDIR = .
UTILDIR = ../utils
BUILDDIR = build

# Source files
SRC = $(SRCDIR)/1_vecadd.cu $(UTILDIR)/data_gen.cu

# Object files
OBJ = $(BUILDDIR)/1_vecadd.o $(BUILDDIR)/data_gen.o

# Output binary
TARGET = vecadd

all: $(TARGET)

# Build target
$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) -o $@ $^

# Compile source files into object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(UTILDIR)/%.cu
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(BUILDDIR) $(TARGET)