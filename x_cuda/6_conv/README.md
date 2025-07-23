# CUDA Convolution Kernel (6_conv)

This project demonstrates how to implement a basic 2D convolution kernel in CUDA, operating on an image loaded via OpenCV.

## What is This?

- Loads an image using the `ImageHandler` utility (`../utils/image_data_gen.h`).
- Prepares the image data for CUDA processing.
- (You will) implement a custom CUDA kernel for 2D convolution.

## How to Build

```bash
make
```

## How to Run

```bash
./6_conv
```

Or use the provided script:

```bash
bash run.sh
```

## Image Input

- The default input image is `../assets/lokiandthor.png`.
- You can change the input by editing the filename in `6_conv.cu`.

## File Structure

- `6_conv.cu` — Main CUDA file, loads image and (to be implemented) runs convolution.
- `Makefile` — For building with `nvcc` and OpenCV.
- `run.sh` — Example script to build and run.

## Next Steps

- Implement your convolution kernel in `6_conv.cu` where marked.
- Use the loaded image data as input and write results back to disk using `ImageHandler`.

--- 