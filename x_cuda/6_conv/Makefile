# compiler
NVCC = nvcc

# compiler flags
CFLAGS = -arch=sm_60 -std=c++11 -I. -I../utils -diag-suppress=611 $(shell pkg-config --cflags opencv4)

LDFLAGS = $(shell pkg-config --libs opencv4)

SRCDIR = .
UTILDIR = ../utils
BUILDDIR = build

SRC = $(SRCDIR)/6_conv.cu $(UTILDIR)/image_data_gen.cc

OBJ = $(BUILDDIR)/6_conv.o $(BUILDDIR)/image_data_gen.o

TARGET = 6_conv

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(UTILDIR)/%.cc
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -rvf $(BUILDDIR) $(TARGET) 