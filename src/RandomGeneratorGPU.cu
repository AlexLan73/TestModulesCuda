#include "RandomGeneratorGPU.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <cmath>

__global__ void applyLowPassConvolutionKernel(int16_t* data, size_t num_points, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        int count = 0;
        for (int i = -half_kernel; i <= half_kernel; ++i) {
            int pos = idx + i;
            if (pos >= 0 && pos < num_points) {
                sum += data[pos];
                count++;
            }
        }
        data[idx] = (int16_t)(sum / count);
    }
}

__global__ void sumKernel(int16_t* data, size_t num_points, int* result) {
    __shared__ int shared_sum[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    shared_sum[local_idx] = (idx < num_points) ? data[idx] : 0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            shared_sum[local_idx] += shared_sum[local_idx + s];
        }
        __syncthreads();
    }
    if (local_idx == 0) {
        atomicAdd(result, shared_sum[0]);
    }
}

__global__ void generateRandomNumbers(int16_t* random_numbers, size_t num_points, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_uniform(&state);
        int16_t num = (int16_t)(floorf(rand_val * 65536.0f)) - 32768;
        random_numbers[idx] = num;
    }
}

RandomGeneratorGPU::RandomGeneratorGPU(unsigned long long seed, size_t num_points)
    : seed_(seed), num_points_(num_points), d_random_numbers_(nullptr) {
    h_random_numbers_.resize(num_points_);
    cudaError_t err = cudaMalloc(&d_random_numbers_, num_points_ * sizeof(int16_t));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
    }
}

RandomGeneratorGPU::~RandomGeneratorGPU() {
    if (d_random_numbers_) {
        cudaFree(d_random_numbers_);
    }
}

void RandomGeneratorGPU::generate() {
    int threads_per_block = 256;
    int blocks = (num_points_ + threads_per_block - 1) / threads_per_block;
    generateRandomNumbers<<<blocks, threads_per_block>>>(d_random_numbers_, num_points_, seed_);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Device synchronize failed");
    }
}

void RandomGeneratorGPU::applyLowPassConvolution(int kernel_size) {
    int threads_per_block = 256;
    int blocks = (num_points_ + threads_per_block - 1) / threads_per_block;
    applyLowPassConvolutionKernel<<<blocks, threads_per_block>>>(d_random_numbers_, num_points_, kernel_size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Low pass convolution synchronize failed");
    }
}

int RandomGeneratorGPU::getSum() const {
    int* d_sum;
    cudaError_t err = cudaMalloc(&d_sum, sizeof(int));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate sum memory");
    }
    err = cudaMemset(d_sum, 0, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_sum);
        throw std::runtime_error("Failed to memset sum");
    }
    int threads_per_block = 256;
    int blocks = (num_points_ + threads_per_block - 1) / threads_per_block;
    sumKernel<<<blocks, threads_per_block>>>(d_random_numbers_, num_points_, d_sum);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_sum);
        throw std::runtime_error("Sum synchronize failed");
    }
    int h_sum;
    err = cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_sum);
        throw std::runtime_error("Failed to copy sum");
    }
    cudaFree(d_sum);
    return h_sum;
}

std::vector<int16_t> RandomGeneratorGPU::getRandomNumbers() const {
    cudaError_t err = cudaMemcpy((void*)h_random_numbers_.data(), d_random_numbers_, num_points_ * sizeof(int16_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy from device to host");
    }
    return h_random_numbers_;
}

void RandomGeneratorGPU::printFirst10() const {
    auto numbers = getRandomNumbers();
    size_t to_print = std::min(static_cast<size_t>(10), num_points_);
    for (size_t i = 0; i < to_print; ++i) {
        std::cout << "Number " << i << ": " << numbers[i] << std::endl;
    }
}