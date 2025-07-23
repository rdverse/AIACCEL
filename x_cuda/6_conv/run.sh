#!/bin/bash

echo "Building and running 6_conv"
rm -rvf conv_output*
make clean
make
./6_conv
rm 6_conv 
