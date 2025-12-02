#include "TestData.hpp"
#include <iostream>

/**
 * @brief Демонстрация использования ООП парсера
 */
void demonstrateUsage() {
    std::cout << "=== OOP JSON Parser Demo ===\n\n";
    
    try {
        // Загружаем оба файла
        std::cout << "Loading test_results.json...\n";
        TestDataContainer data1 = JsonFileManager::loadFromFile("test_results.json");
        
        std::cout << "Loading test_data.json...\n";
        TestDataContainer data2 = JsonFileManager::loadFromFile("test_data.json");
        
        // Выводим информацию
        std::cout << "\n=== FILE 1 ===";
        data1.print();
        
        std::cout << "\n=== FILE 2 ===";
        data2.print();
        
        // Сравниваем
        std::cout << "\n=== COMPARISON ===\n";
        std::cout << data1.compareWith(data2);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

/**
 * @brief Демонстрация модификации данных
 */
void demonstrateModification() {
    std::cout << "\n\n=== Data Modification Demo ===\n\n";
    
    try {
        TestDataContainer data = JsonFileManager::loadFromFile("test_results.json");
        
        // Получаем ссылку на параметры для модификации
        RegFileTFP& tfp = data.getRegFileTFP();
        
        std::cout << "Original N_sig: " << tfp.getNSig() << "\n";
        tfp.setNSig(32);
        std::cout << "Modified N_sig: " << tfp.getNSig() << "\n";
        
        // Работаем с данными
        std::cout << "\nOriginal first element: " << data.getDataElement(0) << "\n";
        data.setDataElement(0, 99.9f);
        std::cout << "Modified first element: " << data.getDataElement(0) << "\n";
        
        // Сохраняем в новый файл
        std::cout << "\nSaving modified data to modified_output.json...\n";
        JsonFileManager::saveToFile(data, "modified_output.json");
        std::cout << "Saved successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

/**
 * @brief Демонстрация работы с отдельными компонентами
 */
void demonstrateComponents() {
    std::cout << "\n\n=== Component Access Demo ===\n\n";
    
    try {
        TestDataContainer data = JsonFileManager::loadFromFile("test_results.json");
        
        // Доступ к компонентам
        const AddrMapHeader& header = data.getAddrMapHeader();
        const RegFileGen& gen = data.getRegFileGen();
        const RegFileTFP& tfp = data.getRegFileTFP();
        
        std::cout << "AddrMapHeader:\n";
        std::cout << "  StrobeNumber: " << header.getStrobeNumber() << "\n";
        std::cout << "  NT: " << header.getNT() << "\n";
        std::cout << "  Start_discrete: " << header.getStartDiscrete() << "\n";
        
        std::cout << "\nRegFileGen:\n";
        std::cout << "  SignalType: " << gen.getSignalType() << "\n";
        
        std::cout << "\nRegFileTFP:\n";
        std::cout << "  gdnum: " << tfp.getGdnum() << "\n";
        std::cout << "  gsnum: " << tfp.getGsnum() << "\n";
        std::cout << "  N_sig: " << tfp.getNSig() << "\n";
        std::cout << "  gsstep is null: " << (tfp.isGsstepNull() ? "yes" : "no") << "\n";
        
        std::cout << "\nDATA Vector:\n";
        std::cout << "  Size: " << data.getDataSize() << " elements\n";
        std::cout << "  Element[0]: " << data.getDataElement(0) << "\n";
        std::cout << "  Element[1]: " << data.getDataElement(1) << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

/**
 * @brief Демонстрация сериализации
 */
void demonstrateSerialization() {
    std::cout << "\n\n=== Serialization Demo ===\n\n";
    
    try {
        // Создаем контейнер с данными
        TestDataContainer data = JsonFileManager::loadFromFile("test_results.json");
        
        // Преобразуем в JSON
        json j = data.toJson();
        
        std::cout << "JSON representation (first 500 chars):\n";
        std::string json_str = j.dump(2);
        std::cout << json_str.substr(0, 500) << "...\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

int main() {
    try {
        // Запускаем все демонстрации
        demonstrateUsage();
        demonstrateModification();
        demonstrateComponents();
        demonstrateSerialization();
        
        std::cout << "\n\n=== All demos completed successfully ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
