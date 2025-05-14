#ifndef simple_funcs
#define simple_funcs

#include <cuda_runtime.h>
#include <iostream>

__host__ void print_gpuspecs(int device_id = 0);

#endif