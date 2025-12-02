#pragma once

#include "JsonSerializable.hpp"
/**
 * @brief Класс для параметров адреса (AddrMapHeader)
 */
class AddrMapHeader : public JsonSerializable {
private:
    int strobeNumber;
    int nt;
    int startDiscrete;

public:
    AddrMapHeader() : strobeNumber(0), nt(0), startDiscrete(0) {}
    
    explicit AddrMapHeader(const json& j) { fromJson(j); }
    
    // Getters
    int getStrobeNumber() const { return strobeNumber; }
    int getNT() const { return nt; }
    int getStartDiscrete() const { return startDiscrete; }
    
    // Setters
    void setStrobeNumber(int value) { strobeNumber = value; }
    void setNT(int value) { nt = value; }
    void setStartDiscrete(int value) { startDiscrete = value; }
    
    // Сравнение
    bool operator==(const AddrMapHeader& other) const {
        return strobeNumber == other.strobeNumber &&
               nt == other.nt &&
               startDiscrete == other.startDiscrete;
    }
    
    void fromJson(const json& j) override {
        strobeNumber = j["StrobeNumber"];
        nt = j["NT"];
        startDiscrete = j["Start_discrete"];
    }
    
    json toJson() const override {
        json j;
        j["StrobeNumber"] = strobeNumber;
        j["NT"] = nt;
        j["Start_discrete"] = startDiscrete;
        return j;
    }
    
    void print(std::ostream& os = std::cout) const override {
        os << "  StrobeNumber: " << strobeNumber << "\n"
           << "  NT: " << nt << "\n"
           << "  Start_discrete: " << startDiscrete << "\n";
    }
};
