#include "Validator.h"
#include <cstring>

// Конструктор
Validator::Validator() : precision_(3), table_width_(80) {

}

// Вывод разделителя
void Validator::printSeparator(int width) {
    std::cout << std::string(width, '-') << "\n";
}

// Вывод заголовка таблицы
void Validator::printTableHeader() {
    std::cout << std::string(table_width_, '-') << "\n";
    std::cout << "| " << std::setw(5) << "Номер"
              << " | " << std::setw(9) << "Веществ."
              << " | " << std::setw(8) << "Мнимая"
              << " | " << std::setw(10) << "Амплитуда"
              << " | " << std::setw(11) << "Фаза (град)"
              << " | " << std::setw(10) << "Энергия"
              << " |\n";
    std::cout << std::string(table_width_, '-') << "\n";
}

// Вывод результатов FFT
void Validator::printFFTResults(const std::vector<std::complex<float>>& spectrum, const char* spectrum_name) {
    std::cout << "Гармоники (энергия " << spectrum_name << "):\n";
    printTableHeader();
    
    for (size_t i = 0; i < spectrum.size(); ++i) {
        float real_part = spectrum[i].real();
        float imag_part = spectrum[i].imag();
        float amplitude = std::abs(spectrum[i]);
        float phase = std::arg(spectrum[i]) * 180.0f / static_cast<float>(M_PI);
        float energy = amplitude * amplitude;
        
        std::cout << "| " << std::setw(5) << i 
                  << " | " << std::scientific << std::setprecision(precision_) << std::setw(9) << real_part
                  << " | " << std::setw(8) << imag_part
                  << " | " << std::fixed << std::setprecision(6) << std::setw(10) << amplitude
                  << " | " << std::setw(11) << phase
                  << " | " << std::scientific << std::setprecision(precision_) << std::setw(10) << energy
                  << " |\n";
    }
    printSeparator(table_width_);
    std::cout << "\n";
  // Найти пик спектра в первом окне (должен быть на частоте 4)
  std::cout << "\n📈 Поиск пика спектра в окне [0]:\n";
  float max_magnitude = 0.0f;
  size_t max_bin = 0;

  auto FFT_SIZE = spectrum.size();

  for (size_t k = 0; k < FFT_SIZE; ++k) {
    float magnitude = std::abs(spectrum[k]);
    if (magnitude > max_magnitude) {
      max_magnitude = magnitude;
      max_bin = k;
    }
  }

  std::cout << "  Максимальная амплитуда на бине: " << max_bin << " (ожидается ~4)\n";
  std::cout << "  Значение амплитуды: " << max_magnitude / FFT_SIZE << "\n\n";
}

// Вывод информации о GPUValidationData
void Validator::printValidationDataInfo(const GPUValidationData& data) {
    std::cout << "GPUValidationData:\n";
    std::cout << "  Размеры: N_gd=" << data.meta.N_gd
              << ", gdnum=" << data.meta.gdnum
              << ", N_sig=" << data.meta.N_sig << "\n";

    if (data.isValid() && data.meta.N_gd > 0 && data.meta.gdnum > 0 && !data.rays.empty() && !data.rays[0].empty()) {
        std::cout << "  Проверка первых 3 точек огибающей [0][0]:\n";
        size_t max_points = std::min(static_cast<size_t>(3), data.rays[0][0].size());
        for (size_t i = 0; i < max_points; ++i) {
            float val = data.rays[0][0][i];
            std::cout << "      [0][0][" << i << "]: " << val << "\n";
        }
    }
    std::cout << "\n";
}

// Сравнение двух спектров
bool Validator::compareSpectra(const std::vector<std::complex<float>>& baseline, 
                                const std::vector<std::complex<float>>& computed,
                                float tolerance) {
    if (baseline.size() != computed.size()) {
        std::cout << "✗ Размеры спектров не совпадают!\n";
        return false;
    }
    
    bool all_match = true;
    for (size_t i = 0; i < baseline.size(); ++i) {
        float diff = std::abs(baseline[i] - computed[i]);
        if (diff > tolerance) {
            std::cout << "✗ Различие на позиции " << i 
                      << ": diff=" << diff << " > tolerance=" << tolerance << "\n";
            all_match = false;
        }
    }
    
    if (all_match) {
        std::cout << "✓ Все точки спектра совпадают (tolerance=" << tolerance << ")\n";
    }
    return all_match;
}

// Валидация результатов из DContext
bool Validator::validateDContextResults(std::string key_name) {
  auto& dcontext = DContext::getInstance();

  if (!dcontext.hasBaseValidationData(key_name)) {
    std::cout << "✗ В DContext нет базовых данных для валидации!\n";
    return false;
  }

  if (!dcontext.hasGPUTestResults(key_name)) {
    std::cout << "✗ В DContext нет данных после GPU для валидации!\n";
    return false;
  }
    
  std::cout << "Проверка данных в DContext...\n";

  auto validation_data = dcontext.getBaseValidationData(key_name);
  auto gpu_test = dcontext.getGPUTestResults(key_name);
    
  if (!validation_data.isValid()) {
    std::cout << "✗ Некорректные данные в ValidationData!\n";
    return false;
  }

  if (!gpu_test.isValid()) {
    std::cout << "✗ Некорректные данные в GPU test!\n";
    return false;
  }
  
  // Сравнить два float вектора
  auto result = VectorComparator::compareFloatVector(validation_data.rays_all, gpu_test.rays_all, 1e-5f, true);

  // Вывести результат
  VectorComparator::printComparisonResult(result, "Test GPU in one vector");
  if (!result.is_valid && result.different_elements <= 20) {
    VectorComparator::printAllDifferences(result, 10);
  }

  if(result.is_valid)
  {
    std::cout << "Тест с GPU прошел!\n";
    return true;
  }

  // Сравнить с точностью epsilon = 1e-5
  auto result1 = VectorComparator::compare3DRays(validation_data.rays, gpu_test.rays, 1e-5f, true);
  VectorComparator::printComparisonResult(result1, "3D rays: базовые vs GPU");
    
  if (!result1.is_valid)
  {
    VectorComparator::printAllDifferences(result1, 5);
    std::cout << "Тест с GPU не прошел!\n";
    return false;

  }

//  std::cout << "✓ Данные валидны!\n";
//  printValidationDataInfo(validation_data);
    
  return true;
}

// Полный отчет о валидации
void Validator::printFullValidationReport(std::string key_name) {
    std::cout << "\n";
    printSeparator(table_width_);
    std::cout << "          ПОЛНЫЙ ОТЧЕТ О ВАЛИДАЦИИ\n";
    printSeparator(table_width_);
    
    // Проверяем наличие данных
    auto& dcontext = DContext::getInstance();
    
    if (dcontext.hasBaseValidationData(key_name)) {
        std::cout << "1. Baseline данные (CPU FFT) для ключа '" << key_name << "':\n";
        auto validation_data = dcontext.getBaseValidationData(key_name);
        printValidationDataInfo(validation_data);
    } else {
        std::cout << "1. Baseline данные для ключа '" << key_name << "': НЕ НАЙДЕНЫ\n";
    }
    
    std::cout << "2. GPU результаты: НАПОМИНАНИЕ - будет реализовано позже\n";
    
    printSeparator(table_width_);
    std::cout << "\n";
}

// НОВЫЕ МЕТОДЫ

// Проверка наличия данных CPU и GPU
bool Validator::checkDataAvailability(bool& has_cpu, bool& has_gpu) {
    auto& dcontext = DContext::getInstance();
    has_cpu = dcontext.hasValidationData();
    
    // TODO: Проверка GPU данных (пока всегда false)
    has_gpu = false;
    
    return has_cpu || has_gpu;
}

// Вывод одного сектора (луч, индекс)
void Validator::printSector(const GPUValidationData& data, int beam, int index, const char* label) {
    std::cout << "Сектор [" << beam << "][" << index << "]";
    if (label && strlen(label) > 0) {
        std::cout << " (" << label << ")";
    } else{
        std::cout << " название сигнала " << data.name << ")";
    }
    std::cout << ":\n";

    if (beam < static_cast<int>(data.rays.size()) && index < static_cast<int>(data.rays[beam].size())) {
        printTableHeader();

        size_t max_points = std::min(static_cast<size_t>(data.meta.N_sig), data.rays[beam][index].size());
        for (size_t i = 0; i < max_points; ++i) {
            float val = data.rays[beam][index][i];
            float amplitude = val; // уже амплитуда
            float energy = amplitude * amplitude;

            std::cout << "| " << std::setw(5) << i
                      << " | " << std::scientific << std::setprecision(precision_) << std::setw(9) << 0.0f // real_part
                      << " | " << std::setw(8) << 0.0f // imag_part
                      << " | " << std::fixed << std::setprecision(6) << std::setw(10) << amplitude
                      << " | " << std::setw(11) << 0.0f // phase
                      << " | " << std::scientific << std::setprecision(precision_) << std::setw(10) << energy
                      << " |\n";
        }
        printSeparator(table_width_);
        std::cout << "\n";
    } else {
        std::cout << "  Сектор недоступен!\n\n";
    }
}

// Сравнение двух секторов
bool Validator::compareSectors(const GPUValidationData& baseline, 
                                const GPUValidationData& computed,
                                int beam, int index,
                                float tolerance) {
  std::cout << "Сравнение сектора [" << beam << "][" << index << "]:\n";
    
  bool all_match = true;
  for (size_t i = 0; i < static_cast<size_t>(baseline.meta.N_sig); ++i) {
    auto base_val = baseline.rays[beam][index][i];
    auto comp_val = computed.rays[beam][index][i];
    float diff = std::abs(base_val - comp_val);
        
    if (diff > tolerance) {
      std::cout << "  ✗ Позиция " << i << ": diff=" << diff << " > tolerance=" << tolerance << "\n";
      all_match = false;
    }
  }
    
  if (all_match) {
    std::cout << "  ✓ Все точки совпадают (tolerance=" << tolerance << ")\n";
  }
    
  return all_match;
}

bool Validator::compareSectors(const GPUValidationData &baseline, const GPUValidationData &computed, float tolerance)
{
  bool all_match = true;

    
  return all_match;
}

// Валидация с конфигурацией
bool Validator::validateWithConfig(const ValidationConfig& config) {
    auto& dcontext = DContext::getInstance();
    
    bool has_cpu, has_gpu;
    checkDataAvailability(has_cpu, has_gpu);
    
    if (!has_cpu && !has_gpu) {
        std::cout << "✗ Нет данных для валидации (ни CPU, ни GPU)\n";
        return false;
    }
    
    if (has_cpu && has_gpu) {
        // Оба источника - сравниваем
        std::cout << "Сравнение CPU и GPU результатов...\n\n";
        
        auto baseline = dcontext.getValidationData();
        // TODO: Получить GPU данные
        auto computed = baseline; // Заглушка
        
        bool all_match = true;
        
        if (config.isFullValidation()) {
            // Валидируем весь сектор
            for (uint32_t b = 0; b < baseline.meta.N_gd; ++b) {
                for (uint32_t i = 0; i < baseline.meta.gdnum; ++i) {
                    if (!compareSectors(baseline, computed, b, i, config.tolerance)) {
                        all_match = false;
                    }
                }
            }
        } else {
            // Валидируем выборочно по конфигурации
            for (const auto& sector : config.sectors) {
                int beam_start = (sector.beam == -1) ? 0 : sector.beam;
                int beam_end = (sector.beam == -1) ? baseline.meta.N_gd : sector.beam + 1;
                int index_start = (sector.index == -1) ? 0 : sector.index;
                int index_end = (sector.index == -1) ? baseline.meta.gdnum : sector.index + 1;
                
                for (int b = beam_start; b < beam_end; ++b) {
                    for (int i = index_start; i < index_end; ++i) {
                        if (!compareSectors(baseline, computed, b, i, config.tolerance)) {
                            all_match = false;
                        }
                    }
                }
            }
        }
        
        printGeneralVerdict(all_match);
        return all_match;
    } else {
        // Только один источник - просто выводим
        if (has_cpu) {
            std::cout << "Только CPU данные (baseline):\n";
            auto baseline = dcontext.getValidationData();
            printValidationDataInfo(baseline);
        }
        if (has_gpu) {
            std::cout << "Только GPU данные: НАПОМИНАНИЕ - будет реализовано\n";
        }
        return true;
    }
}

// Общий вердикт
void Validator::printGeneralVerdict(bool all_match) {
    std::cout << "\n";
    printSeparator(table_width_);
    if (all_match) {
        std::cout << "          ✓ ВАЛИДАЦИЯ ПРОЙДЕНА - Все секторы совпадают!\n";
    } else {
        std::cout << "          ✗ ВАЛИДАЦИЯ ПРОВАЛЕНА - Найдены различия!\n";
    }
    printSeparator(table_width_);
    std::cout << "\n";
}
