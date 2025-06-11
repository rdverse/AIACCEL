#!/bin/bash

#rm -rvf build
echo "RUNNING NVBENCH FOR TILED MATMUL"
mkdir -p build
cd build
cmake ..
make -j$(nproc)
./5_matmul_tiled
rm 5_matmul_tiled
cd -

echo "RUNNING DEBUG TILED MATMUL WITHOUT NVBENCH"
make
./5_matmul_tiled
rm 5_matmul_tiled