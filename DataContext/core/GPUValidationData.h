#pragma once

#include <vector>
#include <complex>
#include <mutex>

/**
 * GPUValidationData - Структура для хранения результатов валидации GPU
 * Формат: mas[луч][index][fft_spectr]
 */
struct GPUValidationData {
  std::string name;                         // название тестируемого сигнала
  TFPBinHeader meta{};                      // метаданные на сигнал    
  std::vector<std::vector<std::vector<float>>> rays;    // сипектр сингала для валидации  mas[луч][index][fft_spectr]         
  std::vector<float> rays_all;              // сипектр сингала для валидации  mas[луч,index,!fft_spectr!]         

  GPUValidationData(TFPBinHeader &meta, std::vector<std::vector<std::vector<float>>> &rays){
    this->meta = meta;
    this->rays = rays;
  }
  GPUValidationData(){  }

  // Проверка что данные валидны
  bool isValid() const {
    return !rays.empty() && meta.N_gd >0 && meta.gdnum > 0 && meta.gsnum > 0  && meta.ray_length > 0;    
  }

  void set_data(TestDatasetNew& dan){
    name = dan.name;
    meta = dan.meta;
    rays = dan.rays;
    rays_all = dan.rays_all;
  }

};
