#pragma once

#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SineGenerator {
private:
    float amplitude;
    float frequency;
    float phase;
    float sample_rate;

public:
    SineGenerator(float amp = 1.0f, float freq = 1.0f, float ph = 0.0f, float sr = 16.0f)
        : amplitude(amp), frequency(freq), phase(ph), sample_rate(sr) {}

    std::vector<float> generate(int num_samples) {
        std::vector<float> signal(num_samples);
        float omega = 2.0f * M_PI * frequency / sample_rate;
        
        for (int i = 0; i < num_samples; ++i) {
            signal[i] = amplitude * sin(omega * i + phase);
        }
        
        return signal;
    }
};
