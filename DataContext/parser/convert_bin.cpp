
#include "convert_bin.h"
#include <cctype>


namespace {
    int find_int_field(const std::string& s, const char* key){
        size_t p = s.find(key); 
        if(p==std::string::npos) return -1; 
        p = s.find(':', p); 
        if(p==std::string::npos) return -1; 
        size_t q = s.find_first_of("0123456789-", p); 
        if(q==std::string::npos) return -1; 
        size_t r=q; 
        while(r<s.size() && (isdigit(s[r])||s[r]=='-')) ++r; 
        return std::stoi(s.substr(q, r-q));
    }
}

namespace d_parser {
  convert_bin::convert_bin(/* args */)  {
  }

  convert_bin::~convert_bin() {
  }

  bool convert_bin::parse_input_json(const std::string &filepath,
                                       std::vector<std::vector<float>> &rays_interleaved,
                                       TFPJsonMeta &meta)
    {
        std::ifstream f(filepath);
        if (!f.is_open())
            return false;
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        meta.N_sig = find_int_field(s, "\"N_sig\"");
        meta.gdnum = find_int_field(s, "\"gdnum\"");
        meta.gdstep = find_int_field(s, "\"gdstep\"");
        meta.gsnum = find_int_field(s, "\"gsnum\"");
        meta.weighting_flag = find_int_field(s, "\"weighting_flag\"");
        if (meta.N_sig <= 0 || meta.gdnum <= 0 || meta.gdstep <= 0 || meta.gsnum <= 0)
            return false;
        size_t pd = s.find("\"DATA\"");
        if (pd == std::string::npos)
            return false;
        pd = s.find('"', pd + 6);
        if (pd == std::string::npos)
            return false;
        size_t pe = s.find('"', pd + 1);
        if (pe == std::string::npos)
            return false;
        std::string data = s.substr(pd + 1, pe - (pd + 1));
        std::vector<float> nums;
        nums.reserve(1 << 20);
        size_t i = 0;
        while (i < data.size())
        {
            while (i < data.size() && (data[i] == ' ' || data[i] == '\n' || data[i] == '\r' || data[i] == '\t'))
                ++i;
            size_t j = i;
            while (j < data.size() && data[j] != ',')
                ++j;
            if (j > i)
                nums.push_back(std::stof(data.substr(i, j - i)));
            i = (j == data.size() ? j : j + 1);
        }
        const size_t rays = 5;
        if (nums.size() % (rays * 2) != 0)
            return false;
        size_t samples = nums.size() / (rays * 2);
        rays_interleaved.assign(rays, std::vector<float>(samples * 2));
        for (size_t idx = 0; idx < samples; ++idx)
        {
            size_t base = idx * rays * 2;
            for (size_t r = 0; r < rays; ++r)
            {
                rays_interleaved[r][2 * idx + 0] = nums[base + 2 * r + 0];
                rays_interleaved[r][2 * idx + 1] = nums[base + 2 * r + 1];
            }
        }
        return true;
    }

  void convert_bin::load_data_json_convert(std::vector<std::string> dirs,     // Директории
            std::vector<std::string> files_data,                               // Исходные данные
            std::vector<std::string> files_results,                            // Тестовые
            std::vector<std::string> files_convert)                            // Данные после конвертации
    
  {
    std::string path_input_data = dirs[0];
    std::string path_output_data = dirs[1];
    std::string path_convert_data = dirs[2];

    // Создать сериализатор
    BinaryDataSerializer serializer;
    
    for(int32_t k0 = 0; k0<files_data.size(); k0++){
//      if(k0==1) break;
      TestDatasetNew meta_dataset_ = TestDatasetNew();

      std::string path_file_in = path_input_data +"\\" + files_data[k0];
      std::string path_file_out = path_output_data +"\\" + files_results[k0];
      std::string path_file_convert = path_convert_data + "\\" +  files_convert[k0];

      convert_one_data(path_file_in, path_file_out, meta_dataset_);
      meta_dataset_.name = std::filesystem::path(files_convert[k0]).stem().string();   // "test5strobe2"

      serializer.writeToFile(path_file_convert, meta_dataset_);
    

      // 2️⃣ ВЕРИФИКАЦИЯ
      auto result = serializer.verifyFile(meta_dataset_, path_file_convert);
      if (result.is_valid) {
          std::cout << "✅ Файл корректен\n";
      }
    
      // 3️⃣ ЧТЕНИЕ данных
      TestDatasetNew loaded;
      serializer.readFromFile(path_file_convert, loaded);
    
      std::cout << "Имя: " << loaded.name << "\n";

    }
       
  }

  void convert_bin::convert_one_data(std::string &file_input_data, std::string &file_validat_data, TestDatasetNew &data_test)
  {
    TestDataContainer data_input_;
    TestDataContainer data_output_;

    try {
      // Загружаем оба файла
      std::cout << "Loading " << file_input_data << "\n";
      data_input_ = JsonFileManager::loadFromFile(file_input_data);

      std::cout << "Loading " << file_validat_data << "\n";
      data_output_ = JsonFileManager::loadFromFile(file_validat_data);
                
    } 
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return;
    }

    auto meta_ = data_input_.getRegFileTFP();
    data_test.meta.gdnum = static_cast<uint32_t>(meta_.getGdnum());
    data_test.meta.gdstep = static_cast<uint32_t>(meta_.getGdstep());
    data_test.meta.gsnum = static_cast<uint32_t>(meta_.getGsnum());
    data_test.meta.N_sig = static_cast<uint32_t>(meta_.getNSig());
    data_test.meta.N_gd = 5;  // Это впеменное использрвание
    data_test.meta.weighting_flag = static_cast<uint32_t>(meta_.getWeightingFlag());

    convert_data(data_input_.getData(), data_test);
    data_test.rays_all = data_output_.getData();
    convert_validation(data_test);
  }

  void convert_bin::convert_data(std::vector<float> &data, TestDatasetNew &data_test)
  {
    auto N_gd = data_test.meta.N_gd;

    // Нужно сохранить промежуточную структуру лучей
    std::vector<std::vector<std::complex<float>>> input_rays;
    input_rays.resize(N_gd);

    data_test.meta.ray_length = data.size() / 2 / data_test.meta.N_gd;
    // Предвыделяем память для каждого луча
    for (auto& ray : input_rays) 
      ray.reserve(data_test.meta.ray_length);

    // Заполняем лучи
    for (size_t idx = 0; idx < data_test.meta.ray_length; ++idx) {
      const size_t base = idx * N_gd * 2;
      for (size_t k = 0; k < N_gd; ++k) {
        float re = data[base + k * 2];
        float im = data[base + k * 2 + 1];
        input_rays[k].emplace_back(re, im);
      }
    }

    // Объединяем в один вектор
    data_test.meta.results_length = data.size() / 2;
//    std::vector<std::complex<float>> input_complex;
    data_test.complex_data.reserve(data_test.meta.results_length);

    for (const auto& ray : input_rays) 
      data_test.complex_data.insert(data_test.complex_data.end(), ray.begin(), ray.end());
            
//    data_test.complex_data = input_complex;

  }
  
  void convert_bin::convert_validation(TestDatasetNew &data_test)
  {
    // Исходные размеры
    const size_t N_gd_ = data_test.meta.N_gd;       // лучей
    const size_t gdnum = data_test.meta.gdnum;      // блоков
    const size_t gsnum = data_test.meta.gsnum;      // гармоник

//    std::vector<std::vector<std::vector<float>>> rays_block_spectr;

    // Инициализация структуры
    data_test.rays.resize(N_gd_);
    for (size_t ray = 0; ray < N_gd_; ++ray) {
      data_test.rays[ray].resize(gdnum);
      for (size_t block = 0; block < gdnum; ++block) {
        data_test.rays[ray][block].resize(gsnum);
      }
    }

    // Копирование данных из flat-вектора
    size_t idx = 0;
    for (size_t ray = 0; ray < N_gd_; ++ray) {
      for (size_t block = 0; block < gdnum; ++block) {
        for (size_t harm = 0; harm < gsnum; ++harm) {
          data_test.rays[ray][block][harm] = data_test.rays_all[idx++];  /* данные */;
        }
      }
    }
//    data_test.rays = rays_block_spectr;     
  }

}