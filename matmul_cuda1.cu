#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 32

__global__ void matMulShared(float *A, float *B, float *C, int N) {

    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {

        int Acol = t * TILE + threadIdx.x;
        if (row < N && Acol < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + Acol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        int Brow = t * TILE + threadIdx.y;
        if (col < N && Brow < N)
            tileB[threadIdx.y][threadIdx.x] = B[Brow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}


int main() {
    int N = 5112;
    size_t size = N * N * sizeof(float);

    float *hA = (float*)malloc(size);
    float *hB = (float*)malloc(size);
    float *hC = (float*)malloc(size);

    for (int i = 0; i < N*N; i++) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

     // tempo Host device
 
    cudaEventRecord(start);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeCopyIn = 0;
    cudaEventElapsedTime(&timeCopyIn, start, stop);

    // configura os blocos
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // tempo do Kernel (memória compartilhada)
   
    cudaEventRecord(start);

    matMulShared<<<blocks, threads>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeKernel = 0;
    cudaEventElapsedTime(&timeKernel, start, stop);

    // tempo device host
    
    cudaEventRecord(start);

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeCopyOut = 0;
    cudaEventElapsedTime(&timeCopyOut, start, stop);

    // resultado
    printf("C[0] = %.1f\n", hC[0]);
    printf("\n===== TEMPOS CUDA Multiplicação de Matrizes Memoria Compartilhada (N = %d) =====\n", N);
    printf("Cópia Host -> Device:  %.3f ms\n", timeCopyIn);
    printf("Kernel Shared:         %.3f ms\n", timeKernel);
    printf("Cópia Device -> Host:  %.3f ms\n", timeCopyOut);
    printf("Tempo TOTAL:           %.3f ms\n", timeCopyIn + timeKernel + timeCopyOut);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
