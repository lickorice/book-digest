#include <iostream>

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n) C[i] = A[i] + B[i];
}

// Compute vector sum d_C = d_A + d_B
void vecAdd(float* A, float* B, float* C, int n) {
    int size = n*sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy values in A, B, and C to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Call kernel function
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    // Copy results back to C
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(void) {
    int N = 1<<20;
    // Initialize float arrays:
    float *A, *B, *C, *expected;
    int size = N*sizeof(float);
    A = (float*) malloc(size);
    B = (float*) malloc(size);
    C = (float*) malloc(size);
    expected = (float*) malloc(size);

    for (int i = 0; i < N; ++i) {
        A[i] = i*5 + i;
        B[i] = i+20;
        expected[i] = A[i] + B[i];
    }

    vecAdd(A, B, C, N);

    int error = 0;
    for (int i = 0; i < N; ++i) {
        error += (expected[i] - C[i]);
    }

    std::cout << "Vector addition error: " << error << std::endl;
}