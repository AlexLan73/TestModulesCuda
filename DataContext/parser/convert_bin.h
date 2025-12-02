#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "JsonFileManager.h"
#include "TestDataContainer.h"
#include "TFPJsonMeta.h"
#include "json_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <TestDataset.h>
#include <TestDatasetNew.h>
#include <filesystem>
#include "BinaryDataSerializer.hpp"

namespace d_parser {
	class convert_bin {
    public:
		convert_bin(/* args */);
    ~convert_bin();
    void load_json_file(std::string path);

    // Парсит входной JSON (TFP_data): формирует 5 лучей, interleaved re,im
    bool parse_input_json(const std::string &filepath,
                        std::vector<std::vector<float>> &rays_interleaved,
                        TFPJsonMeta &meta);


    void load_data_json_convert(std::vector<std::string> dirs,  // Директории
                std::vector<std::string> files_data_,               // Исходные данные
                std::vector<std::string> files_results_,            // Тестовые
                std::vector<std::string> files_convert_             // Данные после конвертации
    );        
    private:
    void convert_one_data(std::string &file_input_data, std::string &file_validat_data, TestDatasetNew &data_test);
    void convert_data(std::vector<float> &data, TestDatasetNew &data_test);
    void convert_validation(TestDatasetNew &data_test);

        

  };

}