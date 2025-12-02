#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matMulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 24000;
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

    // tempo host -> device
    cudaEventRecord(start);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeCopyIn = 0;
    cudaEventElapsedTime(&timeCopyIn, start, stop);

    // configura blocos usando TILE
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // tempo do kernel
    cudaEventRecord(start);
    matMulKernel<<<blocks, threads>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeKernel = 0;
    cudaEventElapsedTime(&timeKernel, start, stop);

    // device -> host
    cudaEventRecord(start);
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeCopyOut = 0;
    cudaEventElapsedTime(&timeCopyOut, start, stop);

    // resultado
    printf("C[0] = %.1f\n", hC[0]);
    printf("\n===== TEMPOS CUDA Multiplicação de Matrizes (N = %d) =====\n", N);
    printf("Cópia Host -> Device:  %.3f ms\n", timeCopyIn);
    printf("Kernel:                %.3f ms\n", timeKernel);
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
