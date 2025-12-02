#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include "JsonSerializable.hpp"
#include "AddrMapHeader.h"
#include "RegFileGen.h"
#include "RegFileTFP.h"


/**
 * @brief Основной класс для работы с данными тестирования
 */
class TestDataContainer : public JsonSerializable {
private:
    AddrMapHeader addrMapHeader;
    RegFileGen regFileGen;
    RegFileTFP regFileTFP;
    std::vector<float> data;
    std::string sourceFile;

public:
    TestDataContainer() = default;
    
    explicit TestDataContainer(const json& j, const std::string& source = "")
        : sourceFile(source) {
        fromJson(j);
    }
    
    // Getters
    const AddrMapHeader& getAddrMapHeader() const { return addrMapHeader; }
    const RegFileGen& getRegFileGen() const { return regFileGen; }
    const RegFileTFP& getRegFileTFP() const { return regFileTFP; }
    const std::vector<float>& getData() const { return data; }
    size_t getDataSize() const { return data.size(); }
    const std::string& getSourceFile() const { return sourceFile; }
    
    // Неконстантные getters
    AddrMapHeader& getAddrMapHeader() { return addrMapHeader; }
    RegFileGen& getRegFileGen() { return regFileGen; }
    RegFileTFP& getRegFileTFP() { return regFileTFP; }
    std::vector<float>& getData() { return data; }
    
    // Получение элемента из DATA по индексу
    float getDataElement(size_t index) const {
        if (index >= data.size()) {
            throw std::out_of_range("Data index out of range");
        }
        return data[index];
    }
    
    // Установка элемента DATA
    void setDataElement(size_t index, float value) {
        if (index >= data.size()) {
            throw std::out_of_range("Data index out of range");
        }
        data[index] = value;
    }
    
    // Сравнение
    bool operator==(const TestDataContainer& other) const {
        return addrMapHeader == other.addrMapHeader &&
               regFileGen == other.regFileGen &&
               regFileTFP == other.regFileTFP &&
               data == other.data;
    }
    
    bool operator!=(const TestDataContainer& other) const {
        return !(*this == other);
    }
    
    void fromJson(const json& j) override {
        addrMapHeader.fromJson(j["AddrMapHeader"]);
        regFileGen.fromJson(j["RegFileGen"]);
        regFileTFP.fromJson(j["RegFileTFP"]);
        
        // Парсим DATA - строка с запятыми
        parseDataString(j["DATA"]);
    }
    
    json toJson() const override {
        json j;
        j["AddrMapHeader"] = addrMapHeader.toJson();
        j["RegFileGen"] = regFileGen.toJson();
        j["RegFileTFP"] = regFileTFP.toJson();
        j["DATA"] = dataToString();
        return j;
    }
    
    void print(std::ostream& os = std::cout) const override {
        if (!sourceFile.empty()) {
            os << "\n=== File: " << sourceFile << " ===\n";
        }
        
        os << "\n=== AddrMapHeader ===\n";
        addrMapHeader.print(os);
        
        os << "\n=== RegFileGen ===\n";
        regFileGen.print(os);
        
        os << "\n=== RegFileTFP ===\n";
        regFileTFP.print(os);
        
        os << "\n=== DATA ===\n"
           << "  Size: " << data.size() << " elements\n";
        
        if (!data.empty()) {
            os << "  First 10 elements: ";
            for (size_t i = 0; i < std::min(size_t(10), data.size()); ++i) {
                os << data[i] << " ";
            }
            os << "\n";
            
            if (data.size() > 20) {
                os << "  Last 10 elements: ";
                for (size_t i = data.size() - 10; i < data.size(); ++i) {
                    os << data[i] << " ";
                }
                os << "\n";
            }
        }
    }
    
    /**
     * @brief Сравнить с другим контейнером и вывести результаты
     */
    std::string compareWith(const TestDataContainer& other) const {
        std::ostringstream oss;
        
        bool identical = true;
        
        // Сравниваем AddrMapHeader
        if (addrMapHeader == other.addrMapHeader) {
            oss << "AddrMapHeader: IDENTICAL\n";
        } else {
            oss << "AddrMapHeader: DIFFERENT\n";
            identical = false;
        }
        
        // Сравниваем RegFileGen
        if (regFileGen == other.regFileGen) {
            oss << "RegFileGen: IDENTICAL\n";
        } else {
            oss << "RegFileGen: DIFFERENT\n";
            identical = false;
        }
        
        // Сравниваем RegFileTFP
        if (regFileTFP == other.regFileTFP) {
            oss << "RegFileTFP: IDENTICAL\n";
        } else {
            oss << "RegFileTFP: DIFFERENT\n";
            identical = false;
        }
        
        // Сравниваем DATA
        if (data.size() != other.data.size()) {
            oss << "DATA: DIFFERENT (size: " << data.size() << " vs " 
                << other.data.size() << ")\n";
            identical = false;
        } else if (data == other.data) {
            oss << "DATA: IDENTICAL (size: " << data.size() << " elements)\n";
        } else {
            oss << "DATA: DIFFERENT (values don't match)\n";
            identical = false;
        }
        
        oss << "\nRESULT: " << (identical ? "ALL DATA IDENTICAL" : "FILES DIFFER") << "\n";
        return oss.str();
    }

private:
    /**
     * @brief Парсинг строки DATA в вектор float
     */
    void parseDataString(const std::string& dataStr) {
        data.clear();
        size_t pos = 0;
        const std::string delimiter = ",";
        
        while (pos < dataStr.length()) {
            size_t end = dataStr.find(delimiter, pos);
            if (end == std::string::npos) {
                end = dataStr.length();
            }
            
            std::string token = dataStr.substr(pos, end - pos);
            if (!token.empty()) {
                try {
                    data.push_back(std::stof(token));
                } catch (const std::exception&) {
                    std::cerr << "Error parsing float value: " << token << "\n";
                }
            }
            
            pos = end + delimiter.length();
        }
    }
    
    /**
     * @brief Преобразование вектора float в строку DATA
     */
    std::string dataToString() const {
        std::ostringstream oss;
        for (size_t i = 0; i < data.size(); ++i) {
            if (i > 0) oss << ",";
            oss << data[i];
        }
        return oss.str();
    }
};
