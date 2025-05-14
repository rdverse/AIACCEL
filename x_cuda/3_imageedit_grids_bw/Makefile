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
SRC = $(SRCDIR)/3_imageedit_grids_bw.cu $(UTILDIR)/simple_funcs.cu $(UTILDIR)/image_data_gen.cc

# object files
OBJ = $(BUILDDIR)/3_imageedit_grids_bw.o $(BUILDDIR)/simple_funcs.o $(BUILDDIR)/image_data_gen.o

TARGET = imageedit_grids_bw

all: $(TARGET)

# Build target
$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

#compile source files into object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(UTILDIR)/%.cu
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Compile C++ source files into object files
$(BUILDDIR)/%.o: $(UTILDIR)/%.cc
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(BUILDDIR) $(TARGET)