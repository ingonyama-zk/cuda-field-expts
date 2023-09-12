#include <iostream>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ __forceinline__ uint32_t add(const uint32_t x, const uint32_t y, const uint32_t modulo) {
    uint32_t s = x + y;
    return s > modulo? s - modulo : s;
}

__device__ __forceinline__ uint32_t mul(const uint32_t x, const uint32_t y, const uint32_t modulo) {
    uint64_t result;
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(result) : "r"(x), "r"(y<<1));
    uint32_t lo;
    uint32_t hi;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r"(hi): "l"(result));
    lo = lo >> 1;
    return add(lo, hi, modulo);
}

__global__ void modularMultiplicationKernel(uint32_t* result, const uint32_t* a, const uint32_t* b, const uint32_t modulo, const int size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > size) { return; }
    result[tid] = mul(a[tid], b[tid], modulo);
}

__global__ void modularAddKernel(int* result, const int* a, const int* b, const int modulo, const int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    result[tid] = add(a[tid], b[tid], modulo);
}


// Host function for modular multiplication
void modularMultiplication(uint32_t* result, const uint32_t* a, const uint32_t* b, const uint32_t modulo, const int size) {
    uint32_t* d_result;
    uint32_t* d_a;
    uint32_t* d_b;

    // Allocate device memory
    gpuErrchk( cudaMalloc((void**)&d_result, size * sizeof(int)) );
    gpuErrchk( cudaMalloc((void**)&d_a, size * sizeof(int)) );

    gpuErrchk( cudaMalloc((void**)&d_b, size * sizeof(int)) );

    // Copy input data from host to device
    gpuErrchk( cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice) );

    // Define block and grid sizes
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch the kernel
    modularMultiplicationKernel<<<numBlocks, blockSize>>>(d_result, d_a, d_b, modulo, size);
    gpuErrchk( cudaGetLastError() );

    // Copy the result from device to host
    gpuErrchk( cudaMemcpy(result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost) );

    // Free device memory
    gpuErrchk( cudaFree(d_result) );
    gpuErrchk( cudaFree(d_a) );
    gpuErrchk( cudaFree(d_b) );
}

// Host function for modular addition
void modularAddition(uint32_t* result, const uint32_t* a, const uint32_t* b, const uint32_t modulo, const int size) {
    int* d_result;
    int* d_a;
    int* d_b;

    // Allocate device memory
    gpuErrchk( cudaMalloc((void**)&d_result, size * sizeof(int)));
    gpuErrchk( cudaMalloc((void**)&d_a, size * sizeof(int)));
    gpuErrchk( cudaMalloc((void**)&d_b, size * sizeof(int)));
    // Copy input data from host to device
    gpuErrchk( cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice));
    // Define block and grid sizes
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch the kernel
    modularAddKernel<<<numBlocks, blockSize>>>(d_result, d_a, d_b, modulo, size);
    gpuErrchk( cudaGetLastError() );

    // Copy the result from device to host
    gpuErrchk( cudaMemcpy(result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost));
    // Free device memory
    gpuErrchk( cudaFree(d_result));
    gpuErrchk( cudaFree(d_a));
    gpuErrchk( cudaFree(d_b));
}


int main() {
    const int size = 2<<29;  // Adjust this size as needed
    const uint32_t modulo = ((uint32_t)(2)<<31) -1;  // The modulo value

    uint32_t* a = new uint32_t[size];
    uint32_t* b = new uint32_t[size];
    uint32_t* result = new uint32_t[size];
    uint32_t* add_result = new uint32_t[size];

    // Initialize your input arrays 'a' and 'b' here

    // for (int i = 0; i < size; i++) {
    //     a[i] = rand();
    //     b[i] = rand();
    //     printf("a[%u] = %u\n", i, a[i]);
    //     printf("b[%u] = %u\n", i, b[i]);
    // }
    
    modularMultiplication(result, a, b, modulo, size);
    modularAddition(add_result, a, b, modulo, size);

    // // Print or use the 'result' array as needed
    // for (int i = 0; i < size; i++) {
    //     printf("result[%u] = %u\n", i, result[i]);
    // }

    //     for (int i = 0; i < size; i++) {
    //     printf("add_result[%u] = %u\n", i, add_result[i]);
    // }

    // // Print the result
    // for (int i = 0; i < size; ++i) {
    //     std::cout << "add_result[" << i << "] = " << add_result[i] << std::endl;
    // }

    // Cleanup host memory
    delete[] a;
    delete[] b;
    delete[] result;
    delete[] add_result;

    return 0;
}