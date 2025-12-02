#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

/**
 * @class VectorComparator
 * @brief Класс для сравнения сложных многомерных векторов с точностью
 * 
 * Поддерживает:
 * - 3D vector<vector<vector<float>>> (rays)
 * - 1D vector<complex<float>> (complex_data)
 * - 1D vector<float> (simple float data)
 * - Детальную диагностику расхождений
 * - Статистику сравнения
 */
class VectorComparator
{
public:
    /**
     * @enum ComparisonStatus
     * @brief Статусы сравнения
     */
    enum class ComparisonStatus
    {
        IDENTICAL = 0,                  // Полностью идентичны
        SIMILAR_WITHIN_TOLERANCE = 1,   // Похожи в пределах точности
        DIFFERENT_SIZE = 2,             // Разные размеры
        VALUES_DIFFER = 3,              // Значения отличаются более чем на epsilon
        EMPTY_VECTOR = 4                // Пустой вектор
    };

    /**
     * @struct DifferenceLocation
     * @brief Местоположение различия в векторе
     */
    struct DifferenceLocation
    {
        size_t index_ray = 0;           // Для 3D: индекс луча
        size_t index_window = 0;        // Для 3D: индекс окна
        size_t index_spectrum = 0;      // Для 3D: индекс спектра
        
        float expected_value = 0.0f;
        float actual_value = 0.0f;
        float difference = 0.0f;
        float relative_error = 0.0f;
    };

    /**
     * @struct ComparisonResult
     * @brief Результат сравнения
     */
    struct ComparisonResult
    {
        ComparisonStatus status = ComparisonStatus::IDENTICAL;
        std::string message;
        bool is_valid = false;
        
        // Статистика
        size_t total_elements = 0;
        size_t different_elements = 0;
        float max_absolute_error = 0.0f;
        float max_relative_error = 0.0f;
        double total_sum_of_squares = 0.0;
        
        // Первое найденное различие
        DifferenceLocation first_difference;
        bool has_difference = false;
        
        // Все найденные различия (если требуется)
        std::vector<DifferenceLocation> all_differences;
    };

    // ===== СРАВНЕНИЕ 3D ВЕКТОРОВ (rays) =====

    /**
     * Сравнить два 3D вектора float с заданной точностью
     */
    static ComparisonResult compare3DRays(
        const std::vector<std::vector<std::vector<float>>>& rays_base,
        const std::vector<std::vector<std::vector<float>>>& rays_test,
        float epsilon = 1e-5f,
        bool collect_all_diffs = false)
    {
        ComparisonResult result;
        result.status = ComparisonStatus::IDENTICAL;
        result.is_valid = true;

        // === Проверка размеров ===
        
        if (rays_base.empty() || rays_test.empty())
        {
            result.status = ComparisonStatus::EMPTY_VECTOR;
            result.message = "❌ Один или оба вектора пусты";
            result.is_valid = false;
            return result;
        }

        if (rays_base.size() != rays_test.size())
        {
            result.status = ComparisonStatus::DIFFERENT_SIZE;
            result.message = "❌ Количество лучей не совпадает: " +
                           std::to_string(rays_base.size()) + " vs " +
                           std::to_string(rays_test.size());
            result.is_valid = false;
            return result;
        }

        // === Сравнение лучей ===
        
        result.total_elements = 0;
        result.different_elements = 0;
        result.max_absolute_error = 0.0f;
        result.max_relative_error = 0.0f;
        result.total_sum_of_squares = 0.0;

        for (size_t i = 0; i < rays_base.size(); ++i)
        {
            if (rays_base[i].size() != rays_test[i].size())
            {
                result.status = ComparisonStatus::DIFFERENT_SIZE;
                result.message = "❌ Луч " + std::to_string(i) +
                               ": количество окон не совпадает: " +
                               std::to_string(rays_base[i].size()) + " vs " +
                               std::to_string(rays_test[i].size());
                result.is_valid = false;
                return result;
            }

            for (size_t j = 0; j < rays_base[i].size(); ++j)
            {
                if (rays_base[i][j].size() != rays_test[i][j].size())
                {
                    result.status = ComparisonStatus::DIFFERENT_SIZE;
                    result.message = "❌ Луч " + std::to_string(i) +
                                   ", окно " + std::to_string(j) +
                                   ": количество спектральных компонент не совпадает: " +
                                   std::to_string(rays_base[i][j].size()) + " vs " +
                                   std::to_string(rays_test[i][j].size());
                    result.is_valid = false;
                    return result;
                }

                for (size_t k = 0; k < rays_base[i][j].size(); ++k)
                {
                    float base_val = rays_base[i][j][k];
                    float test_val = rays_test[i][j][k];
                    float abs_diff = std::abs(base_val - test_val);

                    result.total_elements++;
                    result.total_sum_of_squares += abs_diff * abs_diff;

                    float rel_error = 0.0f;
                    if (std::abs(base_val) > 1e-10f)
                    {
                        rel_error = abs_diff / std::abs(base_val);
                    }

                    if (abs_diff > epsilon)
                    {
                        result.different_elements++;
                        result.max_absolute_error = std::max(result.max_absolute_error, abs_diff);
                        result.max_relative_error = std::max(result.max_relative_error, rel_error);

                        if (!result.has_difference)
                        {
                            result.status = ComparisonStatus::VALUES_DIFFER;
                            result.has_difference = true;
                            result.first_difference.index_ray = i;
                            result.first_difference.index_window = j;
                            result.first_difference.index_spectrum = k;
                            result.first_difference.expected_value = base_val;
                            result.first_difference.actual_value = test_val;
                            result.first_difference.difference = abs_diff;
                            result.first_difference.relative_error = rel_error;
                        }

                        if (collect_all_diffs)
                        {
                            DifferenceLocation loc;
                            loc.index_ray = i;
                            loc.index_window = j;
                            loc.index_spectrum = k;
                            loc.expected_value = base_val;
                            loc.actual_value = test_val;
                            loc.difference = abs_diff;
                            loc.relative_error = rel_error;
                            result.all_differences.push_back(loc);
                        }
                    }
                }
            }
        }

        if (result.different_elements == 0)
        {
            result.status = ComparisonStatus::IDENTICAL;
            result.message = "✅ Все " + std::to_string(result.total_elements) +
                           " элементов идентичны";
            result.is_valid = true;
        }
        else
        {
            result.status = ComparisonStatus::VALUES_DIFFER;
            result.message = "❌ Найдено различий: " +
                           std::to_string(result.different_elements) + " из " +
                           std::to_string(result.total_elements) +
                           " (" + formatPercent(result.different_elements, result.total_elements) + ")";
            result.is_valid = false;
        }

        return result;
    }

    // ===== СРАВНЕНИЕ 1D ВЕКТОРОВ (complex_data) =====

    /**
     * Сравнить два вектора complex<float> с заданной точностью
     */
    static ComparisonResult compareComplex(
        const std::vector<std::complex<float>>& complex_base,
        const std::vector<std::complex<float>>& complex_test,
        float epsilon = 1e-5f,
        bool collect_all_diffs = false)
    {
        ComparisonResult result;
        result.status = ComparisonStatus::IDENTICAL;
        result.is_valid = true;

        if (complex_base.empty() || complex_test.empty())
        {
            result.status = ComparisonStatus::EMPTY_VECTOR;
            result.message = "❌ Один или оба вектора пусты";
            result.is_valid = false;
            return result;
        }

        if (complex_base.size() != complex_test.size())
        {
            result.status = ComparisonStatus::DIFFERENT_SIZE;
            result.message = "❌ Размер вектора не совпадает: " +
                           std::to_string(complex_base.size()) + " vs " +
                           std::to_string(complex_test.size());
            result.is_valid = false;
            return result;
        }

        result.total_elements = complex_base.size();
        result.different_elements = 0;
        result.max_absolute_error = 0.0f;
        result.max_relative_error = 0.0f;
        result.total_sum_of_squares = 0.0;

        for (size_t i = 0; i < complex_base.size(); ++i)
        {
            std::complex<float> base_val = complex_base[i];
            std::complex<float> test_val = complex_test[i];

            std::complex<float> diff = base_val - test_val;
            float abs_diff = std::abs(diff);

            result.total_sum_of_squares += abs_diff * abs_diff;

            float rel_error = 0.0f;
            float base_magnitude = std::abs(base_val);
            if (base_magnitude > 1e-10f)
            {
                rel_error = abs_diff / base_magnitude;
            }

            if (abs_diff > epsilon)
            {
                result.different_elements++;
                result.max_absolute_error = std::max(result.max_absolute_error, abs_diff);
                result.max_relative_error = std::max(result.max_relative_error, rel_error);

                if (!result.has_difference)
                {
                    result.status = ComparisonStatus::VALUES_DIFFER;
                    result.has_difference = true;
                    result.first_difference.index_ray = i;
                    result.first_difference.expected_value = base_val.real();
                    result.first_difference.actual_value = test_val.real();
                    result.first_difference.difference = abs_diff;
                    result.first_difference.relative_error = rel_error;
                }

                if (collect_all_diffs)
                {
                    DifferenceLocation loc;
                    loc.index_ray = i;
                    loc.expected_value = base_val.real();
                    loc.actual_value = test_val.real();
                    loc.difference = abs_diff;
                    loc.relative_error = rel_error;
                    result.all_differences.push_back(loc);
                }
            }
        }

        if (result.different_elements == 0)
        {
            result.status = ComparisonStatus::IDENTICAL;
            result.message = "✅ Все " + std::to_string(result.total_elements) +
                           " элементов идентичны";
            result.is_valid = true;
        }
        else
        {
            result.status = ComparisonStatus::VALUES_DIFFER;
            result.message = "❌ Найдено различий: " +
                           std::to_string(result.different_elements) + " из " +
                           std::to_string(result.total_elements) +
                           " (" + formatPercent(result.different_elements, result.total_elements) + ")";
            result.is_valid = false;
        }

        return result;
    }

    // ===== СРАВНЕНИЕ 1D ВЕКТОРОВ (float data) =====

    /**
     * Сравнить два 1D вектора float с заданной точностью
     * @param data_base Эталонный вектор [индекс]
     * @param data_test Тестируемый вектор [индекс]
     * @param epsilon Допустимая абсолютная ошибка (по умолчанию 1e-5f)
     * @param collect_all_diffs Собрать ВСЕ различия
     * @return Результат сравнения с диагностикой
     */
    static ComparisonResult compareFloatVector(
        const std::vector<float>& data_base,
        const std::vector<float>& data_test,
        float epsilon = 1e-5f,
        bool collect_all_diffs = false)
    {
        ComparisonResult result;
        result.status = ComparisonStatus::IDENTICAL;
        result.is_valid = true;

        // === Проверка размеров ===

        if (data_base.empty() || data_test.empty())
        {
            result.status = ComparisonStatus::EMPTY_VECTOR;
            result.message = "❌ Один или оба вектора пусты";
            result.is_valid = false;
            return result;
        }

        if (data_base.size() != data_test.size())
        {
            result.status = ComparisonStatus::DIFFERENT_SIZE;
            result.message = "❌ Размер вектора не совпадает: " +
                           std::to_string(data_base.size()) + " vs " +
                           std::to_string(data_test.size());
            result.is_valid = false;
            return result;
        }

        // === Сравнение элементов ===

        result.total_elements = data_base.size();
        result.different_elements = 0;
        result.max_absolute_error = 0.0f;
        result.max_relative_error = 0.0f;
        result.total_sum_of_squares = 0.0;

        for (size_t i = 0; i < data_base.size(); ++i)
        {
            float base_val = data_base[i];
            float test_val = data_test[i];
            float abs_diff = std::abs(base_val - test_val);

            result.total_sum_of_squares += abs_diff * abs_diff;

            // Вычислить относительную ошибку
            float rel_error = 0.0f;
            if (std::abs(base_val) > epsilon)
            {
                rel_error = abs_diff / std::abs(base_val);
            }

            // Проверить превышение ошибки
            if (abs_diff > epsilon)
            {
                result.different_elements++;
                result.max_absolute_error = std::max(result.max_absolute_error, abs_diff);
                result.max_relative_error = std::max(result.max_relative_error, rel_error);

                // Сохранить информацию о первом различии
                if (!result.has_difference)
                {
                    result.status = ComparisonStatus::VALUES_DIFFER;
                    result.has_difference = true;
                    result.first_difference.index_ray = i;
                    result.first_difference.expected_value = base_val;
                    result.first_difference.actual_value = test_val;
                    result.first_difference.difference = abs_diff;
                    result.first_difference.relative_error = rel_error;
                }

                // Если требуется, собрать ВСЕ различия
                if (collect_all_diffs)
                {
                    DifferenceLocation loc;
                    loc.index_ray = i;
                    loc.expected_value = base_val;
                    loc.actual_value = test_val;
                    loc.difference = abs_diff;
                    loc.relative_error = rel_error;
                    result.all_differences.push_back(loc);
                }
            }
        }

        // === Формирование сообщения результата ===

        if (result.different_elements == 0)
        {
            result.status = ComparisonStatus::IDENTICAL;
            result.message = "✅ Все " + std::to_string(result.total_elements) +
                           " элементов идентичны";
            result.is_valid = true;
        }
        else
        {
            result.status = ComparisonStatus::VALUES_DIFFER;
            result.message = "❌ Найдено различий: " +
                           std::to_string(result.different_elements) + " из " +
                           std::to_string(result.total_elements) +
                           " (" + formatPercent(result.different_elements, result.total_elements) + ")";
            result.is_valid = false;
        }

        return result;
    }

    // ===== ВЫВОД РЕЗУЛЬТАТОВ =====

    /**
     * Вывести результат сравнения в консоль
     */
    static void printComparisonResult(const ComparisonResult& result, const std::string& title = "")
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        if (!title.empty())
            std::cout << title << "\n";
        std::cout << "📊 РЕЗУЛЬТАТ СРАВНЕНИЯ\n";
        std::cout << std::string(80, '=') << "\n";

        std::cout << "Статус: ";
        switch (result.status)
        {
            case ComparisonStatus::IDENTICAL:
                std::cout << "✅ ИДЕНТИЧНЫ\n";
                break;
            case ComparisonStatus::SIMILAR_WITHIN_TOLERANCE:
                std::cout << "⚠️  ПОХОЖИ В ПРЕДЕЛАХ ТОЧНОСТИ\n";
                break;
            case ComparisonStatus::DIFFERENT_SIZE:
                std::cout << "❌ РАЗНЫЕ РАЗМЕРЫ\n";
                break;
            case ComparisonStatus::VALUES_DIFFER:
                std::cout << "❌ ЗНАЧЕНИЯ ОТЛИЧАЮТСЯ\n";
                break;
            case ComparisonStatus::EMPTY_VECTOR:
                std::cout << "❌ ПУСТОЙ ВЕКТОР\n";
                break;
        }

        std::cout << "Сообщение: " << result.message << "\n";

        if (result.total_elements > 0)
        {
            std::cout << "\n📈 СТАТИСТИКА:\n";
            std::cout << "  Всего элементов: " << result.total_elements << "\n";
            std::cout << "  Разных элементов: " << result.different_elements << "\n";
            std::cout << "  Максимальная абсолютная ошибка: " << std::scientific 
                      << result.max_absolute_error << "\n";
            std::cout << "  Максимальная относительная ошибка: " 
                      << (result.max_relative_error * 100.0f) << "%\n";
            std::cout << "  RMS (Root Mean Square): " << std::scientific 
                      << std::sqrt(result.total_sum_of_squares / result.total_elements) << "\n";
        }

        if (result.has_difference)
        {
            std::cout << "\n🔍 ПЕРВОЕ РАЗЛИЧИЕ:\n";
            printDifferenceLocation(result.first_difference);
        }

        std::cout << std::string(80, '=') << "\n\n";
    }

    /**
     * Вывести все найденные различия
     */
    static void printAllDifferences(const ComparisonResult& result, size_t max_to_print = 10)
    {
        if (result.all_differences.empty())
        {
            std::cout << "✅ Различий не найдено\n";
            return;
        }

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "📋 ВСЕ РАЗЛИЧИЯ (показано " << 
                     std::min(max_to_print, result.all_differences.size()) << 
                     " из " << result.all_differences.size() << ")\n";
        std::cout << std::string(80, '=') << "\n";

        for (size_t idx = 0; idx < std::min(max_to_print, result.all_differences.size()); ++idx)
        {
            std::cout << "\n[" << (idx + 1) << "]:\n";
            printDifferenceLocation(result.all_differences[idx]);
        }

        if (result.all_differences.size() > max_to_print)
        {
            std::cout << "\n... и ещё " << (result.all_differences.size() - max_to_print)
                      << " различий\n";
        }

        std::cout << std::string(80, '=') << "\n\n";
    }

private:
    /**
     * Вывести информацию о различии
     */
    static void printDifferenceLocation(const DifferenceLocation& loc)
    {
        if (loc.index_window == 0 && loc.index_spectrum == 0 && loc.index_ray != 0)
        {
            // Это 1D вектор
            std::cout << "  Индекс: [" << loc.index_ray << "]\n";
        }
        else if (loc.index_spectrum == 0 && loc.index_window != 0)
        {
            // Это 2D вектор
            std::cout << "  Позиция: [" << loc.index_ray << "][" 
                      << loc.index_window << "]\n";
        }
        else if (loc.index_spectrum != 0)
        {
            // Это 3D вектор
            std::cout << "  Позиция: [луч:" << loc.index_ray 
                      << "][окно:" << loc.index_window 
                      << "][спектр:" << loc.index_spectrum << "]\n";
        }

        std::cout << "  Ожидалось: " << std::scientific << loc.expected_value << "\n";
        std::cout << "  Получено: " << std::scientific << loc.actual_value << "\n";
        std::cout << "  Абсолютная разница: " << std::scientific << loc.difference << "\n";
        std::cout << "  Относительная ошибка: " << (loc.relative_error * 100.0f) << "%\n";
    }

    /**
     * Форматировать процент
     */
    static std::string formatPercent(size_t part, size_t total)
    {
        if (total == 0) return "0%";
        double percent = (static_cast<double>(part) / total) * 100.0;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << percent << "%";
        return ss.str();
    }
};
