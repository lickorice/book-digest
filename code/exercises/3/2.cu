// Matrix-vector multiplication program for square matrices
// Note that matrices are represented in a single-dimension vector
#include <iostream>

// For item b:
__global__ void matVecMul(float* A, float* B, float* C, int N) {
    if (threadIdx.x < N) {
        float mulSum = 0;
        for (int i = 0; i < N; ++i) {
            int idx = (threadIdx.x * N) + i;
            mulSum += (B[idx] * C[i]);
        }
        A[threadIdx.x] = mulSum;
    }
}

void matMulStub(float* A, float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    int size = N*sizeof(float);

    // Allocate:
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, N*size); // B is the square matrix 
    cudaMalloc((void**)&d_C, size);

    // Copy
    cudaMemcpy(d_B, B, N*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    matVecMul<<<1, N>>>(d_A, d_B, d_C, N); // Each thread processes a row of B

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
    int size = N*sizeof(float);

    // Allocate:
    A = (float*) malloc(size); 
    B = (float*) malloc(N*size); // B is the square matrix 
    C = (float*) malloc(size); 
    expected = (float*) malloc(size); 

    // Populate B and C:
    for (int i = 0; i < N; ++i) {
        C[i] = 1.1;
        for (int j = 0; j < N; ++j) {
            int idx = (i*N) + j;
            B[idx] = 2.2;
        }
    }

    // Get expected values:
    for (int i = 0; i < N; ++i) {
        float matVecSum = 0;
        for (int j = 0; j < N; ++j) {
            int idx = (i*N) + j;
            matVecSum += (B[idx] * C[j]);
            // std::cout << B[idx] << " " << C[j] << std::endl;
        }
        expected[i] = matVecSum;
    }

    matMulStub(A, B, C, N);

    // Check error:
    float error = 0;
    for (int i = 0; i < N; ++i) {
        error += (A[i] - expected[i]);
        // std::cout << A[i] << " " << expected[i] << std::endl;
    }
    std::cout << "Matrix-vector multiplication error: " << error << std::endl;
}