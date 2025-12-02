#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include <algorithm>
#include "SignalData.h"
#include "GPUValidationData.h"

/**
 * @class DContext
 * @brief Singleton для хранения нескольких сигналов и валидационных данных
 * 
 * Архитектура:
 * - map_signal_data: SignalData (данные с диска, ключ = name)
 * - map_base_validation_data: GPUValidationData (базовые данные для валидации)
 * - map_gpu_test_results: GPUValidationData (результаты тестов на GPU)
 * 
 * Потокобезопасность: все операции защищены мьютексом
 */
class DContext
{
public:
    // === SINGLETON INTERFACE ===
    
    static DContext& getInstance();
    
    DContext(const DContext&) = delete;
    DContext& operator=(const DContext&) = delete;

#pragma region   SignalData  // ===== MAP 1: SignalData (данные с диска) =====

    // ===== MAP 1: SignalData (данные с диска) =====
    
    /**
     * Добавить или обновить сигнал данные
     * @param name Ключ (название сигнала)
     * @param data Данные сигнала
     */
    void addSignalData(const std::string& name, const SignalData& data);
    
    /**
     * Получить сигнал данные по ключу
     * @param name Ключ (название сигнала)
     * @return SignalData или пустой объект если не найден
     */
    SignalData getSignalData(const std::string& name) const;
    
    /**
     * Получить все сигнал данные
     */
    std::map<std::string, SignalData> getAllSignalData() const;
    
    /**
     * Проверить наличие сигнала по ключу
     */
    bool hasSignalData(const std::string& name) const;
    
    /**
     * Удалить сигнал данные по ключу
     */
    bool removeSignalData(const std::string& name);
    
    /**
     * Получить количество сигналов в хранилище
     */
    size_t getSignalDataCount() const;
    
    /**
     * Очистить все сигнал данные
     */
    void clearAllSignalData();
#pragma endregion   SignalData

#pragma region   GPUValidationData  // ===== MAP 2: GPUValidationData (базовые данные для валидации) =====  
    // ===== MAP 2: GPUValidationData (базовые данные для валидации) =====
    
    /**
     * Добавить или обновить базовые валидационные данные
     * @param name Ключ (название тестируемого сигнала)
     * @param data Валидационные данные
     */
    void addBaseValidationData(const std::string& name, const GPUValidationData& data);
    
    /**
     * Получить базовые валидационные данные по ключу
     */
    GPUValidationData getBaseValidationData(const std::string& name) const;

    /**
     * Получить первые базовые валидационные данные (для обратной совместимости)
     */
    GPUValidationData getBaseValidationData() const;
    
    /**
     * Получить все базовые валидационные данные
     */
    std::map<std::string, GPUValidationData> getAllBaseValidationData() const;
    
    /**
     * Проверить наличие базовых данных по ключу
     */
    bool hasBaseValidationData(const std::string& name) const;

    /**
     * Проверить наличие любых базовых данных
     */
    bool hasBaseValidationData() const;

    // Синонимы для обратной совместимости
    bool hasValidationData() const;
    GPUValidationData getValidationData() const;
    
    /**
     * Удалить базовые валидационные данные по ключу
     */
    bool removeBaseValidationData(const std::string& name);
    
    /**
     * Получить количество валидационных наборов
     */
    size_t getBaseValidationDataCount() const;
    
    /**
     * Очистить все базовые валидационные данные
     */
    void clearAllBaseValidationData();

#pragma endregion GPUValidationData

#pragma region   GPUTestResults  // ===== MAP 3: GPUValidationData (результаты тестов GPU) =====
    // ===== MAP 3: GPUValidationData (результаты тестов GPU) =====
    
    /**
     * Добавить или обновить результаты тестов GPU
     * @param name Ключ (название тестируемого сигнала)
     * @param data Результаты тестов
     */
    void addGPUTestResults(const std::string& name, const GPUValidationData& data);
    
    /**
     * Получить результаты тестов GPU по ключу
     */
    GPUValidationData getGPUTestResults(const std::string& name) const;
    
    /**
     * Получить все результаты тестов GPU
     */
    std::map<std::string, GPUValidationData> getAllGPUTestResults() const;
    
    /**
     * Проверить наличие результатов тестов по ключу
     */
    bool hasGPUTestResults(const std::string& name) const;
    
    /**
     * Удалить результаты тестов по ключу
     */
    bool removeGPUTestResults(const std::string& name);
    
    /**
     * Получить количество результатов тестов
     */
    size_t getGPUTestResultsCount() const;
    
    /**
     * Очистить все результаты тестов
     */
    void clearAllGPUTestResults();
#pragma endregion  GPUTestResults

    // ===== ОБЩИЕ МЕТОДЫ =====
    
    /**
     * Получить все ключи из всех map-ов
     */
    std::vector<std::string> getAllKeys() const;
    
    /**
     * Очистить все данные
     */
    void clearAllData();
    
    /**
     * Получить статистику хранилища
     */
    struct StorageStats
    {
        size_t signal_data_count = 0;
        size_t base_validation_count = 0;
        size_t gpu_test_results_count = 0;
        size_t total_count = 0;
    };
    
    StorageStats getStorageStats() const;
    
    /**
     * Вывести статистику в консоль
     */
    void printStorageStats() const;

private:
    DContext() = default;
    ~DContext() = default;

    // === ПРИВАТНЫЕ ДАННЫЕ ===
    
    // MAP 1: Сигнал данные (читаются с диска)
    std::map<std::string, SignalData> map_signal_data_;
    
    // MAP 2: Базовые валидационные данные (эталон для тестирования)
    std::map<std::string, GPUValidationData> map_base_validation_data_;
    
    // MAP 3: Результаты тестов GPU (для сравнения и валидации)
    std::map<std::string, GPUValidationData> map_gpu_test_results_;
    
    // Мьютекс для потокобезопасности
    mutable std::mutex mutex_;
};
