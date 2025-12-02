#include "json_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>

static int find_int_field(const std::string& s, const char* key){
    size_t p = s.find(key); if(p==std::string::npos) return -1; p = s.find(':', p); if(p==std::string::npos) return -1; size_t q = s.find_first_of("0123456789-", p); if(q==std::string::npos) return -1; size_t r=q; while(r<s.size() && (isdigit(s[r])||s[r]=='-')) ++r; return std::stoi(s.substr(q, r-q));
}

bool parse_input_json(const std::string& filepath,
                      std::vector<std::vector<float>>& rays_interleaved,
                      TFPJsonMeta& meta)
{
    std::ifstream f(filepath); if(!f.is_open()) return false;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    meta.N_sig = find_int_field(s, "\"N_sig\"");
    meta.gdnum = find_int_field(s, "\"gdnum\"");
    meta.gdstep= find_int_field(s, "\"gdstep\"");
    meta.gsnum = find_int_field(s, "\"gsnum\"");
    meta.weighting_flag = find_int_field(s, "\"weighting_flag\"");
    if(meta.N_sig<=0||meta.gdnum<=0||meta.gdstep<=0||meta.gsnum<=0) return false;
    size_t pd = s.find("\"DATA\""); if(pd==std::string::npos) return false;
    pd = s.find('"', pd+6); if(pd==std::string::npos) return false; size_t pe = s.find('"', pd+1); if(pe==std::string::npos) return false;
    std::string data = s.substr(pd+1, pe-(pd+1));
    std::vector<float> nums; nums.reserve(1<<20);
    size_t i=0; while(i<data.size()) { while(i<data.size() && (data[i]==' '||data[i]=='\n'||data[i]=='\r'||data[i]=='\t')) ++i; size_t j=i; while(j<data.size() && data[j]!=',') ++j; if(j>i) nums.push_back(std::stof(data.substr(i, j-i))); i = (j==data.size()? j : j+1); }
    const size_t rays = 5;
    if(nums.size()%(rays*2)!=0) return false;
    size_t samples = nums.size()/(rays*2);
    rays_interleaved.assign(rays, std::vector<float>(samples*2));
    for(size_t idx=0; idx<samples; ++idx){
        size_t base = idx * rays * 2;
        for(size_t r=0; r<rays; ++r){
            rays_interleaved[r][2*idx+0] = nums[base + 2*r + 0];
            rays_interleaved[r][2*idx+1] = nums[base + 2*r + 1];
        }
    }
    return true;
}


