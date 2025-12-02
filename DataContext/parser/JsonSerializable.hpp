#pragma once

#include <iostream>
#include "../nlohmann/json.hpp"

using json = nlohmann::json;

/**
 * @brief Базовый класс для JSON сериализации
 */
class JsonSerializable {
public:
    virtual ~JsonSerializable() = default;
    
    /**
     * @brief Парсинг из JSON объекта
     */
    virtual void fromJson(const json& j) = 0;
    
    /**
     * @brief Преобразование в JSON объект
     */
    virtual json toJson() const = 0;
    
    /**
     * @brief Вывод информации
     */
    virtual void print(std::ostream& os = std::cout) const = 0;
};


