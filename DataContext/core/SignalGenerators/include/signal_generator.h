#pragma once

#include <vector>
#include <complex>
#include <string>
#include <DContext.h>
#include <SignalData.h>

// Forward declaration


// Компонент сигнала (для многокомпонентных сигналов)
struct SignalComponent {
    float amplitude = 1.0f;
    int period = 16;
    float phase = 0.0f;
};

// Структура конфигурации для генерации
struct SignalConfig {
    // Базовые параметры
    int period = 16;
    int num_samples = 4096;
    float amplitude = 1.0f;
    float phase = 0.0f;
    
    // Расширенные параметры для сложных сигналов
    std::string type = "simple";  // "simple", "multi_sine", "sine_noise", "am_modulation"
    
    // Для многокомпонентных сигналов (несколько синусоид)
    std::vector<SignalComponent> components;
    
    // Для сигналов с шумом
    float noise_amplitude = 0.0f;
    std::string noise_type = "white";  // "white", "pink"
    
    // Для модулированных сигналов
    SignalComponent carrier;
    SignalComponent modulator;
};

/**
 * SignalGenerator - Singleton для генерации комплексных сигналов
 * Генерирует сигналы и записывает их в DContext
 */
class SignalGenerator {
public:
    std::string name_sig ="sin_test";

    static SignalGenerator& getInstance();
    
    SignalGenerator(const SignalGenerator&) = delete;
    SignalGenerator& operator=(const SignalGenerator&) = delete;

    /**
     * Генерация: 5 лучей на 4096 точек
     * Генерирует и автоматически сохраняет в DContext
     */
    void test_5_4096([[maybe_unused]] int period, std::string name_sig="sin_test");

    /**
     * Генерация: 5 лучей, размер увеличен в 4 раза на 4096 точек
     * Генерирует и автоматически сохраняет в DContext
     */
    void test_5_4_4096([[maybe_unused]] int period, std::string name_sig="sin_test");

    /**
     * Генерация синусоидального сигнала
     * fft16 для 3D OpenCl [16][800][5] fft16 на 800 гипотиз 5 лучей
     * Генерирует и автоматически сохраняет в DContext
     */
    void test_16_800_5([[maybe_unused]] int period=8, std::string name_sig="sin_test");

    /**
     * Генерация синусоидального сигнала
     * fft32 для 3D OpenCl [32][400][5] fft32 на 400 гипотиз 5 лучей
     * Генерирует и автоматически сохраняет в DContext
     */
    void test_32_400_5([[maybe_unused]] int period=16, std::string name_sig="sin_test");

    /**
     * Генерация синусоидального сигнала
     * fft64 для 3D OpenCl [64][200][5] fft32 на 400 гипотиз 5 лучей
     * Генерирует и автоматически сохраняет в DContext
     */
    void test_64_200_5([[maybe_unused]] int period=32, std::string name_sig="sin_test");

    /**
     * Генерация синусоидального сигнала
     * num_samples - обязательно
     * period - обязательно
     * amplitude = 1 по умолчанию
     * phase = 0 по умолчанию
     * Генерирует и автоматически сохраняет в DContext
     */
    void generateSine(int num_samples, int period, float amplitude = 1.0f, float phase = 0.0f);

    /**
     * Генерация из конфигурации
     * Генерирует и автоматически сохраняет в DContext
     */
    void generate_from_json(const SignalConfig& config);

    /**
     * Генерация локализованной синусоиды в заданном месте
     * @param vector_length - длина комплексного вектора (1024)
     * @param start_index - индекс начала синуса (64)
     * @param duration - длительность синуса (64 точки)
     * @param period - период синусоиды (32 точки)
     * @param amplitude - амплитуда (1.0)
     * @param phase - фаза (0.0)
     * Автоматически обрезает, если выходит за границы
     */
    void generateLocalizedSine(int vector_length, int start_index, int duration, 
                                int period, float amplitude = 1.0f, float phase = 0.0f);

private:
    SignalGenerator() = default;
    ~SignalGenerator() = default;
    
    SignalData signal_data_;
    // Вспомогательный метод для генерации и сохранения
    void generateAndSave([[maybe_unused]] const std::vector<std::complex<float>>& signal);
    void generateTestSignal(int num_beams, int samples_per_beam, int period, int n_signal, std::string name_sig, int beam_multiplier = 1);
    // Инициализация  
    DContext& ctx = DContext::getInstance();

    SignalData initial_signal_data() const
    {
        SignalData signal_data = SignalData();
        signal_data.name = "sin_test";
        signal_data.meta.gdnum = 765;
        signal_data.meta.gdstep =2;
        signal_data.meta.gsnum = 21;
        signal_data.meta.N_gd = 5;
        signal_data.meta.N_sig = 8;
        signal_data.meta.version = 2;
        signal_data.meta.weighting_flag = 0;
        return signal_data;
    }

}; 