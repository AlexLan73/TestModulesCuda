

#pragma once
#include <iostream>
#include "RegFileTFP.h"
#include "TestDataContainer.h"
#include "JsonFileManager.h"


/**
 * @brief Класс для сравнения файлов
 */
class FileComparator {
public:
    /**
     * @brief Сравнить два JSON файла
     */
    static void compare(const std::string& file1, const std::string& file2) {
        std::cout << "\n=== COMPARING FILES ===\n";
        
        TestDataContainer data1 = JsonFileManager::loadFromFile(file1);
        TestDataContainer data2 = JsonFileManager::loadFromFile(file2);
        
        data1.print();
        data2.print();
        
        std::cout << "\n" << data1.compareWith(data2);
    }
};
