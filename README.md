# Project 3 Parallel Computing
## Dependencies
To compile and run this program you need to have the following dependencies installed:
- Nvidia CUDA Toolkit: [https://docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- LibJPEG: sudo apt install libjpeg-dev
## Compilation
Use the following to compile the programs:
- all programs:
``` bash
make
```
- hough base:
``` bash
make hough
```
- hough with constant memory:
``` bash
make houghConstant
```
- hough with constant and shared memory:
``` bash
make houghShared
```
## Execution
To run the programs use the following commands:

- hough base:
``` bash
./hough [image.pgm]
```
- hough with constant memory:
``` bash
./houghConstant [image.pgm]
```
- hough with constant and shared memory:
``` bash
./houghShared [image.pgm]
```

Replace `[image.pgm]` with the filename of the image you want to analize.
## Authors
- Sebastián Maldonado Arnau 18003
- Alexis Renato Estrada Martinez 181099
- Roberto Alejandro Castillo de Leon 18546
