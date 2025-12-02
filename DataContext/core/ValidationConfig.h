#pragma once

#include <vector>
#include <string>

/**
 * Параметры валидации для одного сектора
 */
struct SectorConfig {
    int beam;      // Номер луча (-1 = все лучи)
    int index;     // Номер индекса (-1 = все индексы)
    
    SectorConfig() : beam(-1), index(-1) {}
    SectorConfig(int b, int i) : beam(b), index(i) {}
};

/**
 * Конфигурация валидации
 * JSON формат:
 * {
 *   "sectors": [
 *     {"beam": 0, "index": 3},
 *     {"beam": 1, "index": 2},
 *     {"beam": -1, "index": -1}  // -1 = все
 *   ]
 * }
 * Если sectors не указан или пуст - валидируем весь сектор
 */
struct ValidationConfig {
    std::vector<SectorConfig> sectors;  // Список секторов для валидации
    float tolerance;                     // Допустимая погрешность
    bool show_details;                   // Показывать детали
    
    ValidationConfig() : tolerance(1e-5f), show_details(true) {}
    
    // Проверка: валидировать весь сектор?
    bool isFullValidation() const {
        return sectors.empty();
    }
    
    // Проверка: валидировать конкретный луч?
    bool hasBeam(int beam) const {
        if (isFullValidation()) return true;
        for (const auto& s : sectors) {
            if (s.beam == beam || s.beam == -1) return true;
        }
        return false;
    }
    
    // Проверка: валидировать конкретный индекс в луче?
    bool hasIndex(int beam, int index) const {
        if (isFullValidation()) return true;
        for (const auto& s : sectors) {
            // Точное совпадение
            if (s.beam == beam && s.index == index) return true;
            // В этом луче - все индексы
            if (s.beam == beam && s.index == -1) return true;
            // Все лучи, но этот индекс
            if (s.beam == -1 && s.index == index) return true;
            // Все лучи и все индексы
            if (s.beam == -1 && s.index == -1) return true;
        }
        return false;
    }
};
