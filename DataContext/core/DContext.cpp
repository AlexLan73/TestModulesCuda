#include "DContext.h"
#include <iostream>
#include <iomanip>

// ===== SINGLETON =====

DContext& DContext::getInstance()
{
    static DContext instance;
    return instance;
}

// ===== MAP 1: SignalData (данные с диска) =====

void DContext::addSignalData(const std::string& name, const SignalData& data)
{
    std::lock_guard<std::mutex> lock(mutex_);
    map_signal_data_[name] = data;
    
    if (!data.name.empty() && data.name != name)
    {
        std::cout << "⚠️  Внутреннее имя SignalData отличается от ключа\n";
    }
}

SignalData DContext::getSignalData(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = map_signal_data_.find(name);
    if (it != map_signal_data_.end())
    {
        return it->second;
    }
    
    return SignalData();
}

std::map<std::string, SignalData> DContext::getAllSignalData() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return map_signal_data_;
}

bool DContext::hasSignalData(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return map_signal_data_.find(name) != map_signal_data_.end();
}

bool DContext::removeSignalData(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = map_signal_data_.find(name);
    if (it != map_signal_data_.end())
    {
        map_signal_data_.erase(it);
        return true;
    }
    
    return false;
}

size_t DContext::getSignalDataCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return map_signal_data_.size();
}

void DContext::clearAllSignalData()
{
    std::lock_guard<std::mutex> lock(mutex_);
    map_signal_data_.clear();
}

// ===== MAP 2: GPUValidationData (базовые данные для валидации) =====

void DContext::addBaseValidationData(const std::string& name, const GPUValidationData& data)
{
    std::lock_guard<std::mutex> lock(mutex_);
    map_base_validation_data_[name] = data;
    
    if (!data.name.empty() && data.name != name)
    {
        std::cout << "⚠️  Внутреннее имя базовых валидационных данных отличается от ключа\n";
    }
}

GPUValidationData DContext::getBaseValidationData(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = map_base_validation_data_.find(name);
    if (it != map_base_validation_data_.end())
    {
        return it->second;
    }

    return GPUValidationData();
}

GPUValidationData DContext::getBaseValidationData() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!map_base_validation_data_.empty())
    {
        return map_base_validation_data_.begin()->second;
    }

    return GPUValidationData();
}

std::map<std::string, GPUValidationData> DContext::getAllBaseValidationData() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return map_base_validation_data_;
}

bool DContext::hasBaseValidationData(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return map_base_validation_data_.find(name) != map_base_validation_data_.end();
}

bool DContext::hasBaseValidationData() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return !map_base_validation_data_.empty();
}

// Синонимы для обратной совместимости
bool DContext::hasValidationData() const
{
    return hasBaseValidationData();
}

GPUValidationData DContext::getValidationData() const
{
    return getBaseValidationData();
}

bool DContext::removeBaseValidationData(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = map_base_validation_data_.find(name);
    if (it != map_base_validation_data_.end())
    {
        map_base_validation_data_.erase(it);
        return true;
    }
    
    return false;
}

size_t DContext::getBaseValidationDataCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return map_base_validation_data_.size();
}

void DContext::clearAllBaseValidationData()
{
    std::lock_guard<std::mutex> lock(mutex_);
    map_base_validation_data_.clear();
}

// ===== MAP 3: GPUValidationData (результаты тестов GPU) =====

void DContext::addGPUTestResults(const std::string& name, const GPUValidationData& data)
{
    std::lock_guard<std::mutex> lock(mutex_);
    map_gpu_test_results_[name] = data;
    
    if (!data.name.empty() && data.name != name)
    {
        std::cout << "⚠️  Внутреннее имя результатов GPU отличается от ключа\n";
    }
}

GPUValidationData DContext::getGPUTestResults(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = map_gpu_test_results_.find(name);
    if (it != map_gpu_test_results_.end())
    {
        return it->second;
    }
    
    return GPUValidationData();
}

std::map<std::string, GPUValidationData> DContext::getAllGPUTestResults() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return map_gpu_test_results_;
}

bool DContext::hasGPUTestResults(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return map_gpu_test_results_.find(name) != map_gpu_test_results_.end();
}

bool DContext::removeGPUTestResults(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = map_gpu_test_results_.find(name);
    if (it != map_gpu_test_results_.end())
    {
        map_gpu_test_results_.erase(it);
        return true;
    }
    
    return false;
}

size_t DContext::getGPUTestResultsCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return map_gpu_test_results_.size();
}

void DContext::clearAllGPUTestResults()
{
    std::lock_guard<std::mutex> lock(mutex_);
    map_gpu_test_results_.clear();
}

// ===== ОБЩИЕ МЕТОДЫ =====

std::vector<std::string> DContext::getAllKeys() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> keys;
    
    for (const auto& pair : map_signal_data_)
        keys.push_back(pair.first);
    
    for (const auto& pair : map_base_validation_data_)
    {
        if (std::find(keys.begin(), keys.end(), pair.first) == keys.end())
            keys.push_back(pair.first);
    }
    
    for (const auto& pair : map_gpu_test_results_)
    {
        if (std::find(keys.begin(), keys.end(), pair.first) == keys.end())
            keys.push_back(pair.first);
    }
    
    return keys;
}

void DContext::clearAllData()
{
    std::lock_guard<std::mutex> lock(mutex_);
    map_signal_data_.clear();
    map_base_validation_data_.clear();
    map_gpu_test_results_.clear();
}

DContext::StorageStats DContext::getStorageStats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    StorageStats stats;
    stats.signal_data_count = map_signal_data_.size();
    stats.base_validation_count = map_base_validation_data_.size();
    stats.gpu_test_results_count = map_gpu_test_results_.size();
    stats.total_count = stats.signal_data_count + stats.base_validation_count + 
                        stats.gpu_test_results_count;
    
    return stats;
}

void DContext::printStorageStats() const
{
    auto stats = getStorageStats();
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "📊 СТАТИСТИКА ХРАНИЛИЩА DCONTEXT\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "📦 SignalData:             " << std::setw(5) << stats.signal_data_count << "\n";
    std::cout << "🔍 Базовые валидационные: " << std::setw(5) << stats.base_validation_count << "\n";
    std::cout << "🎯 Результаты GPU тестов: " << std::setw(5) << stats.gpu_test_results_count << "\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "📊 ВСЕГО:                 " << std::setw(5) << stats.total_count << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}
