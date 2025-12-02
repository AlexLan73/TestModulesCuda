#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "TFPJsonMeta.h"

// Парсит входной JSON (TFP_data): формирует 5 лучей, interleaved re,im
bool parse_input_json(const std::string &filepath,
                      std::vector<std::vector<float>> &rays_interleaved,
                      TFPJsonMeta &meta);
