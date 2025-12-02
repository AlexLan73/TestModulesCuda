#pragma once

#include "JsonSerializable.hpp"


/**
 * @brief Класс для параметров генератора (RegFileGen)
 */
class RegFileGen : public JsonSerializable {
private:
    int signalType;

public:
    RegFileGen() : signalType(0) {}
    
    explicit RegFileGen(const json& j) { fromJson(j); }
    
    // Getters
    int getSignalType() const { return signalType; }
    
    // Setters
    void setSignalType(int value) { signalType = value; }
    
    // Сравнение
    bool operator==(const RegFileGen& other) const {
        return signalType == other.signalType;
    }
    
    void fromJson(const json& j) override {
        signalType = j["SignalType"];
    }
    
    json toJson() const override {
        json j;
        j["SignalType"] = signalType;
        return j;
    }
    
    void print(std::ostream& os = std::cout) const override {
        os << "  SignalType: " << signalType << "\n";
    }
};
