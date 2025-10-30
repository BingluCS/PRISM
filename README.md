# PRISM: An Efficient GPU-Based Lossy Compression Framework for Progressive Data Retrieval with Multi-Level Interpolation

PRISM is an error-controlled lossy compressor that supports progressive data retrieval for floating-point data (both single- and double-precision).

PRISM offers several key features:

1. 🚀 **Progressive decompression**: a single compressed file natively supports decompression at multiple precision levels.
2. 🧩**Incremental Loading**: allows the decompressed data to be incrementally reconstructed during retrieval to satisfy diverse error requirements.
3. 📊**High compression ratio**: achieves one of the best compression ratios among state-of-the-art (SOTA) GPU-based lossy compressors for floating-point data.
4. ⚡ **High throughpu**t: delivers **2–19x** higher compression/decompression throughput (~100 GB/s) compared to other progressive compressors.

## Environment Requirements

- Linux OS with NVIDIA GPUs
- Git >= 2.15
- CMake >= 3.23
- CUDA Toolkit >= 11.0
- GCC >= 7.3.0

## Compile and Use PRISM

You can compile and install PRISM with the following commands.

```
git clone https://github.com/BingluCS/PRISM.git
cd PRISM
mkdir build && cd build
cmake ..
make -j
```

After installation, you will see executable binary generated in ```PRISM/build/```

## Usage

Details on how to use PRISM can be found below.

```
Options:
  -i <path> : Path to binary input file containing the original data
  -x <path> : Path to decompressed binary output file (optional)
  -z <path> : Path to compressed binary output/input file (optional)
  -f : single precision (float type)
  -d : double precision (double type)
  -1 <nx> : Dimensions for 1D array a[nx]
  -2 <nx> <ny> : Dimensions for 2D array a[ny][nx]
  -3 <nx> <ny> <nz> : Dimensions for 3D array a[nz][ny][nx]
  -4 <nx> <ny> <nz> <nw> : Dimensions for 4D array a[nw][nz][ny][nx]
  -A [errorBound] : Absolute error bound
  -R [errorBound] : Relative error bound
  --report [time,cr] : output the compression ratio/time
  -prog : Enable progressive compression/decompression
  -errors -<nums> <error_1> <error_2> ... : Perform progressive decompression multiple times with gradually decreasing error bounds
Examples:
  ./prism -i [oriFilePath] -f -3 [dim_z] [dim_y] [dim_x] -R [errorBound] -z [cmpFilePath] -x [decFilePath] --report time,cr
  ./prism -i [oriFilePath] -d -3 [dim_z] [dim_y] [dim_x] -R [errorBound] -z [cmpFilePath] -x [decFilePath] -prog -errors -2 1e-1 1e-2
```

Example Commands:

- non-progressive mode

  ```
  ./prism -i Miranda/density.d64 -d -3 384 384 256 -R 1E-6 -z -x --report time,cr
  ```

- progressive mode

  ```
  ./prism -i Miranda/density.d64 -d -3 384 384 256 -R 1E-6 -z -x -prog -errors -5 1e-2 1e-3 1e-4 1e-5 1e-6 --report time,cr
  ```
