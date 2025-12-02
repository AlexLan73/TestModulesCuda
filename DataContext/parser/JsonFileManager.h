#pragma once

#include <fstream>
#include <iterator>
#include <stdexcept>
#include "JsonSerializable.hpp"
#include "AddrMapHeader.h"
#include "RegFileGen.h"
#include "TestDataContainer.h"

/**
 * @brief Класс для работы с JSON файлами
 */
class JsonFileManager {
public:
    /**
     * @brief Загрузить данные из JSON файла
     */
    static TestDataContainer loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        file.close();
        
        json j = json::parse(content);
        
        return TestDataContainer(j, filename);
    }
    
    /**
     * @brief Сохранить данные в JSON файл
     */
    static void saveToFile(const TestDataContainer& container, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create file: " + filename);
        }
        
        file << container.toJson().dump(4) << "\n";
        file.close();
    }
};
