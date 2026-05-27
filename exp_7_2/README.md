# How to run

```
$ source env.sh
$ echo $NEUWARE_HOME
$ which cncc
$ whicn cnas
$ bash test.sh
```

# Reference baseline

cntoolkit version:

```
$ cat /usr/local/neuware/version.txt
Neuware Version 3.15.11
```

## MLU370-M8@1300MHz & Hygon C86 7285 32-core Processor

cnmon info:

```
    Product Name                   : MLU370-M8
    Firmware                       : v1.1.6
    Driver                         : v5.10.26
    Compute Mode                   : Default
```

test log:

```
+ rm -rf build
+ mkdir build
+ cncc --bang-arch=compute_30 -O3 01_scalar.mlu -o build/01_scalar
+ ./build/01_scalar

M = 128, N = 256, K = 256

CPU Time taken: 12.396 ms

MLU Time taken: 5280.594 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 02_scalar_nram.mlu -o build/02_scalar_nram
+ ./build/02_scalar_nram

M = 128, N = 256, K = 256

CPU Time taken: 11.749 ms

MLU Time taken: 193.940 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 03_vector_nram.mlu -o build/03_vector_nram
+ ./build/03_vector_nram

M = 128, N = 256, K = 256

CPU Time taken: 11.801 ms

MLU Time taken: 0.061 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 04_vector_nram_blocks.mlu -o build/04_vector_nram_blocks
+ ./build/04_vector_nram_blocks

M = 8192, N = 512, K = 512

CPU Time taken: 23081.127 ms

MLU Time taken: 0.453 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 05_vector_nram_blocks_pipe3.mlu -o build/05_vector_nram_blocks_pipe3
+ ./build/05_vector_nram_blocks_pipe3

M = 8192, N = 512, K = 512

CPU Time taken: 23196.375 ms

MLU Time taken: 0.444 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 06_vector_sram_unions_pipe5.mlu -o build/06_vector_sram_unions_pipe5
+ ./build/06_vector_sram_unions_pipe5

M = 8192, N = 512, K = 512

CPU Time taken: 23094.615 ms

MLU Time taken: 0.373 ms

PASSED
```
