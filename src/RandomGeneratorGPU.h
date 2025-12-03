#ifndef RANDOM_GENERATOR_GPU_H
#define RANDOM_GENERATOR_GPU_H

#include <vector>
#include <cstdint>

class RandomGeneratorGPU {
public:
    RandomGeneratorGPU(unsigned long long seed, float max_random, size_t num_points);
    ~RandomGeneratorGPU();

    void generate();
    void applyLowPassConvolution(int kernel_size);
    int getSum() const;
    std::vector<int16_t> getRandomNumbers() const;
    void printFirst10() const;

private:
    unsigned long long seed_;
    size_t num_points_;
    int16_t* d_random_numbers_;
    float max_random_;
    std::vector<int16_t> h_random_numbers_;
};

#endif // RANDOM_GENERATOR_GPU_H