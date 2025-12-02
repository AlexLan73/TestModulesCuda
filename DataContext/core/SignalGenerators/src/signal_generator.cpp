#include "signal_generator.h"
#include "../../DataContext/core/DContext.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SignalGenerator &SignalGenerator::getInstance()
{
    static SignalGenerator instance;
    return instance;
}

void SignalGenerator::generateTestSignal(int num_beams, int samples_per_beam, int period, int n_signal, std::string name_sig, int beam_multiplier)
{
    this->name_sig = name_sig;
    const int total_size = num_beams * beam_multiplier * samples_per_beam;
    std::vector<std::complex<float>> result(total_size);

    float omega = 2.0f * static_cast<float>(M_PI) / static_cast<float>(period);

    for (int beam = 0; beam < num_beams; ++beam)
    {
        for (int multi = 0; multi < beam_multiplier; ++multi)
        {
            for (int i = 0; i < samples_per_beam; ++i)
            {
                float value = sinf(omega * i);
                int index = (beam * beam_multiplier + multi) * samples_per_beam + i;
                result[index] = std::complex<float>(value, 0.0f);
            }
        }
    }

    signal_data_ = initial_signal_data();
    signal_data_.data = result;
    signal_data_.name = this->name_sig;
    signal_data_.meta.results_length = result.size();
    signal_data_.meta.ray_length = signal_data_.meta.results_length / num_beams;
    signal_data_.meta.N_sig = n_signal;
    signal_data_.meta.gsnum = 21;
    ctx.addSignalData(this->name_sig, signal_data_);
}

void SignalGenerator::test_5_4096([[maybe_unused]] int period, std::string name_sig)
{
    generateTestSignal(5, 4096, period, 16, name_sig, 1);
}

void SignalGenerator::test_5_4_4096([[maybe_unused]] int period, std::string name_sig)
{
    generateTestSignal(5, 4096, period, 16, name_sig, 4);
}

void SignalGenerator::generateSine(int num_samples, int period, float amplitude, float phase)
{
    std::vector<std::complex<float>> result(num_samples);
    float omega = 2.0f * static_cast<float>(M_PI) / static_cast<float>(period);

    for (int i = 0; i < num_samples; ++i)
    {
        float value = amplitude * sinf(omega * i + phase);
        result[i] = std::complex<float>(value, 0.0f);
    }

    // Сохраняем в DContext
    generateAndSave(result);
}

void SignalGenerator::generate_from_json(const SignalConfig &config)
{
    // Используем параметры из конфигурации
    generateSine(config.num_samples, config.period, config.amplitude, config.phase);
}

void SignalGenerator::generateLocalizedSine(int vector_length, int start_index, int duration,
                                            int period, float amplitude, float phase)
{
    // Создаем вектор нужной длины, заполненный нулями
    std::vector<std::complex<float>> result(vector_length, std::complex<float>(0.0f, 0.0f));

    // Вычисляем параметры синуса
    float omega = 2.0f * static_cast<float>(M_PI) / static_cast<float>(period);

    // Определяем границы (обрезаем, если выходит за пределы)
    int end_index = start_index + duration;
    if (start_index < 0)
        start_index = 0;
    if (end_index > vector_length)
        end_index = vector_length;
    if (end_index <= start_index)
    {
        // Нет места для синуса - сохраняем нулевой вектор
        generateAndSave(result);
        return;
    }

    // Генерируем синус только в заданном диапазоне
    for (int i = start_index; i < end_index; ++i)
    {
        float value = amplitude * sinf(omega * (i - start_index) + phase);
        result[i] = std::complex<float>(value, 0.0f);
    }

    // Сохраняем в DContext
    generateAndSave(result);
}

void SignalGenerator::generateAndSave([[maybe_unused]] const std::vector<std::complex<float>> &signal)
{
}

void SignalGenerator::test_16_800_5([[maybe_unused]] int period, std::string name_sig)
{
    generateTestSignal(5, 800 * 16, period, 8, name_sig, 1);
}
void SignalGenerator::test_32_400_5([[maybe_unused]] int period, std::string name_sig)
{
    generateTestSignal(5, 400 * 32, period, 16, name_sig, 1);
}
void SignalGenerator::test_64_200_5([[maybe_unused]] int period, std::string name_sig)
{
    generateTestSignal(5, 200 * 64, period, 32, name_sig, 1);
}
