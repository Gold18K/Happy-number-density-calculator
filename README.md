A CUDA accelerated program that empirically calculates the density of happy numbers of the set of integers of n digits.

To use it (Windows):

1) Create a new CUDA project on Visual Studio 2022 (Requires CUDA Toolkit with Visual Studio support), and add all files in "main" folder to it.
2) Compile.
3) Run.

To use it (Linux):

1) Open console in the "main" folder.
2) Compile using command (Requires CUDA Toolkit): nvcc Main.cu Kernels.cu Cuda_Utilities.cpp -o cuda_happy
3) Run using command: ./cuda_happy
