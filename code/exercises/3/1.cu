// Matrix addition program for square matrices
// Note that matrices are represented in a single-dimension vector
#include <iostream>

// For item b:
__global__ void matAdd_b(float* A, float* B, float* C, int N) {
    // Block processes row, thread processes element:
    if (blockIdx.x < N && threadIdx.x < N) {
        int idx = (blockIdx.x * N) + threadIdx.x;
        A[idx] = B[idx] + C[idx];
    }
}

void matAdd(float* A, float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    int size = N*N*sizeof(float);

    // Allocate:
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    matAdd_b<<<N, N>>>(d_A, d_B, d_C, N);

    // Copy
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    // Free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(void) {
    int N = 1 << 10; // Dimensions of the matrix
    float *A, *B, *C, *expected;
    int size = N*N*sizeof(float);

    // Allocate:
    A = (float*) malloc(size); 
    B = (float*) malloc(size); 
    C = (float*) malloc(size); 
    expected = (float*) malloc(size); 

    // Populate B and C:
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = (i*N) + j;
            B[idx] = i*j + i + j + 4;
            C[idx] = (i+1) * (j+2);
            expected[idx] = B[idx] + C[idx];
        }
    }

    matAdd(A, B, C, N);

    // Check error:
    int error = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = (i*N) + j;
            error += A[idx] - expected[idx];
            // std::cout << A[idx] << " " << expected[idx] << std::endl;
        }
    }
    std::cout << "Matrix addition error: " << error << std::endl;
}