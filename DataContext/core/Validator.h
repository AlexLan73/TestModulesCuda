#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include "DContext.h"
#include "ValidationConfig.h"
#include "VectorComparator.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Validator - Класс для валидации и сравнения результатов FFT
 * Сравнивает baseline (GPUValidationData) с результатами GPU
 */
class Validator {
public:
    // Конструктор
    Validator();
    
    // Деструктор
    ~Validator() = default;
    
    // Вывод результатов FFT (форматированная таблица)
    void printFFTResults(const std::vector<std::complex<float>>& spectrum, const char* spectrum_name);
    
    // Вывод информации о GPUValidationData
    void printValidationDataInfo(const GPUValidationData& data);
    
    // Сравнение двух спектров с допустимой погрешностью
    bool compareSpectra(const std::vector<std::complex<float>>& baseline, 
                        const std::vector<std::complex<float>>& computed,
                        float tolerance = 1e-5f);

    // Валидация результатов из DContext
    bool validateDContextResults(std::string key_name);
    // Полный вывод для валидации (красивое форматирование)
    void printFullValidationReport(std::string key_name);
    
    // НОВЫЕ МЕТОДЫ для выборочной валидации
    // Вывод одного сектора (луч, индекс)
    void printSector(const GPUValidationData& data, int beam, int index, const char* label = "");
    
    // Сравнение двух секторов (baseline vs computed)
    bool compareSectors(const GPUValidationData& baseline, 
                        const GPUValidationData& computed,
                        int beam, int index,
                        float tolerance = 1e-5f);

    // Сравнение двух спектров (baseline vs computed)
    bool compareSectors(const GPUValidationData& baseline, 
                        const GPUValidationData& computed,
                        float tolerance = 1e-5f);
                        
    // Валидация с конфигурацией
    bool validateWithConfig(const ValidationConfig& config);
    
    // Общий вердикт: все совпадают или нет
    void printGeneralVerdict(bool all_match);
    
    // Проверка наличия данных CPU и GPU
    bool checkDataAvailability(bool& has_cpu, bool& has_gpu);

private:
  // Вспомогательные методы
  void printSeparator(int width = 80);
  void printTableHeader();
    
  // Настройки вывода
  int precision_;
  int table_width_;
  

};
