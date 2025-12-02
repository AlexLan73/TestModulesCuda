
#pragma once

#include "JsonSerializable.hpp"
#include "AddrMapHeader.h"
#include "RegFileGen.h"

/**
 * @brief Класс для параметров трансформации (RegFileTFP)
 */
class RegFileTFP : public JsonSerializable {
private:
    int gd1;
    int gdstep;
    int gdnum;
    int gsnum;
    int gs1;
    bool gsstepIsNull;
    double gsstepValue;
    int nSig;
    int weightingFlag;
    int porCode;
    int dumpChvo;

public:
    RegFileTFP() 
        : gd1(0), gdstep(0), gdnum(0), gsnum(0), gs1(0),
          gsstepIsNull(true), gsstepValue(0.0), nSig(0),
          weightingFlag(0), porCode(0), dumpChvo(0) {}
    
    explicit RegFileTFP(const json& j) { fromJson(j); }
    
    // Getters
    int getGd1() const { return gd1; }
    int getGdstep() const { return gdstep; }
    int getGdnum() const { return gdnum; }
    int getGsnum() const { return gsnum; }
    int getGs1() const { return gs1; }
    bool isGsstepNull() const { return gsstepIsNull; }
    double getGsstepValue() const { return gsstepValue; }
    int getNSig() const { return nSig; }
    int getWeightingFlag() const { return weightingFlag; }
    int getPorCode() const { return porCode; }
    int getDumpChvo() const { return dumpChvo; }
    
    // Setters
    void setGd1(int value) { gd1 = value; }
    void setGdstep(int value) { gdstep = value; }
    void setGdnum(int value) { gdnum = value; }
    void setGsnum(int value) { gsnum = value; }
    void setGs1(int value) { gs1 = value; }
    void setGsstep(double value) { gsstepValue = value; gsstepIsNull = false; }
    void setGsstepNull(bool isNull) { gsstepIsNull = isNull; }
    void setNSig(int value) { nSig = value; }
    void setWeightingFlag(int value) { weightingFlag = value; }
    void setPorCode(int value) { porCode = value; }
    void setDumpChvo(int value) { dumpChvo = value; }
    
    // Сравнение
    bool operator==(const RegFileTFP& other) const {
        return gd1 == other.gd1 &&
               gdstep == other.gdstep &&
               gdnum == other.gdnum &&
               gsnum == other.gsnum &&
               gs1 == other.gs1 &&
               gsstepIsNull == other.gsstepIsNull &&
               (gsstepIsNull || gsstepValue == other.gsstepValue) &&
               nSig == other.nSig &&
               weightingFlag == other.weightingFlag &&
               porCode == other.porCode &&
               dumpChvo == other.dumpChvo;
    }
    
    void fromJson(const json& j) override {
        gd1 = j["gd1"];
        gdstep = j["gdstep"];
        gdnum = j["gdnum"];
        gsnum = j["gsnum"];
        gs1 = j["gs1"];
        
        if (j["gsstep"].is_null()) {
            gsstepIsNull = true;
            gsstepValue = 0.0;
        } else {
            gsstepIsNull = false;
            gsstepValue = j["gsstep"];
        }
        
        nSig = j["N_sig"];
        weightingFlag = j["weighting_flag"];
        porCode = j["por_code"];
        dumpChvo = j["dump_chvo"];
    }
    
    json toJson() const override {
        json j;
        j["gd1"] = gd1;
        j["gdstep"] = gdstep;
        j["gdnum"] = gdnum;
        j["gsnum"] = gsnum;
        j["gs1"] = gs1;
        j["gsstep"] = gsstepIsNull ? json() : json(gsstepValue);        
        //j["gsstep"] = gsstepIsNull ? json::value_t::null : json(gsstepValue);
        j["N_sig"] = nSig;
        j["weighting_flag"] = weightingFlag;
        j["por_code"] = porCode;
        j["dump_chvo"] = dumpChvo;
        return j;
    }
    
    void print(std::ostream& os = std::cout) const override {
        os << "  gd1: " << gd1 << "\n"
           << "  gdstep: " << gdstep << "\n"
           << "  gdnum: " << gdnum << "\n"
           << "  gsnum: " << gsnum << "\n"
           << "  gs1: " << gs1 << "\n"
           << "  gsstep: " << (gsstepIsNull ? "null" : std::to_string(gsstepValue)) << "\n"
           << "  N_sig: " << nSig << "\n"
           << "  weighting_flag: " << weightingFlag << "\n"
           << "  por_code: " << porCode << "\n"
           << "  dump_chvo: " << dumpChvo << "\n";
    }
};

