# 📡 SignalGenerator - Инструкция по подключению

**Дата создания:** 2025-01-27  
**Автор:** Кодо (AI Assistant)

---

## 📋 Что было создано:

### 1. **DContext** (DataContext/core/)
Singleton для хранения сгенерированных сигналов.

**Файлы:**
- `DataContext/core/DContext.h`
- `DataContext/core/DContext.cpp`
- `DataContext/CMakeLists.txt`

### 2. **SignalGenerator** (Shared/SignalGenerators/)
Singleton для генерации комплексных сигналов.

**Файлы:**
- `Shared/SignalGenerators/include/signal_generator.h`
- `Shared/SignalGenerators/src/signal_generator.cpp`

### 3. **Структура конфигурации:**
```cpp
struct SignalConfig {
    int period = 16;        // период сигнала (обязательно)
    int num_samples = 4096; // количество точек (обязательно)
    float amplitude = 1.0f; // амплитуда (по умолчанию 1.0)
    float phase = 0.0f;     // фаза (по умолчанию 0.0)
};
```

---

## 🚀 Быстрый старт

### Шаг 1: Включите библиотеки в ваш CMakeLists.txt

```cmake
# 1. DataContext library
add_subdirectory(DataContext)

# 2. SignalGenerators library  
add_subdirectory(Shared)

# 3. Ваш исполняемый файл
add_executable(your_program your_main.cpp)

# 4. Свяжите библиотеки
target_link_libraries(your_program 
    SignalGenerators 
    DataContext
)
```

### Шаг 2: Используйте в коде

```cpp
#include "Shared/SignalGenerators/include/signal_generator.h"
#include "DataContext/core/DContext.h"

int main() {
    // Получаем экземпляры
    auto& generator = SignalGenerator::getInstance();
    auto& dcontext = DContext::getInstance();
    
    // Вариант 1: Простая генерация
    generator.generateSine(1024, 8);
    
    // Вариант 2: Через конфигурацию
    SignalConfig config;
    config.period = 8;
    config.num_samples = 1024;
    config.amplitude = 2.0f;
    generator.generate_from_json(config);
    
    // Читаем из DContext
    if (dcontext.hasSignal()) {
        auto signal = dcontext.getLastSignal();
        // Используйте signal...
    }
    
    return 0;
}
```

---

## 📊 Доступные методы генерации:

### 1. `test_5_4096(int period)`
Генерирует 5 лучей по 4096 точек.
**Размер выходного вектора:** 20,480 точек

```cpp
generator.test_5_4096(16);
```

### 2. `test_5_4_4096(int period)`
Генерирует 20 лучей (5×4) по 4096 точек.
**Размер выходного вектора:** 81,920 точек

```cpp
generator.test_5_4_4096(16);
```

### 3. `generateSine(num_samples, period, amplitude, phase)`
Генерирует настраиваемую синусоиду.

```cpp
generator.generateSine(1024, 8, 1.0f, 0.0f);
```

### 4. `generate_from_json(SignalConfig)`
Генерирует из конфигурации.

```cpp
SignalConfig config;
config.period = 8;
config.num_samples = 1024;
generator.generate_from_json(config);
```

---

## 🔧 Что важно знать:

### ✓ Формат данных:
- Все методы генерируют `std::vector<std::complex<float>>`
- Данные автоматически сохраняются в `DContext`
- Все сигналы комплексные (сейчас только real часть)

### ✓ DContext методы:
```cpp
void setLastSignal(const std::vector<std::complex<float>>& signal);
std::vector<std::complex<float>> getLastSignal() const;
bool hasSignal() const;
void clearSignal();
```

### ✓ Thread Safety:
DContext использует `std::mutex` для потокобезопасности.

---

## 📁 Структура проекта:

```
AmdOpenCLTest01/
├── DataContext/
│   ├── CMakeLists.txt
│   └── core/
│       ├── DContext.h
│       └── DContext.cpp
│
├── Shared/
│   ├── CMakeLists.txt (обновлен!)
│   └── SignalGenerators/
│       ├── include/
│       │   └── signal_generator.h
│       └── src/
│           └── signal_generator.cpp
│
└── test_fft_chain.cpp (пример использования)
```

---

## ⚠️ Важные изменения в CMakeLists.txt:

### `Shared/CMakeLists.txt` должен содержать:

```cmake
# DataContext dependency
add_subdirectory(${CMAKE_SOURCE_DIR}/DataContext)

# SignalGenerators library
add_library(SignalGenerators STATIC
    SignalGenerators/src/signal_generator.cpp
)

target_include_directories(SignalGenerators PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/SignalGenerators/include
    ${CMAKE_SOURCE_DIR}/DataContext/core
)

target_link_libraries(SignalGenerators PUBLIC DataContext)
```

---

## 🧪 Примеры тестов:

Смотрите `test_fft_chain.cpp` для полного примера:
1. Генерация сигнала
2. Чтение из DContext
3. FFT обработка
4. Анализ гармоник

---

## ✅ Что уже протестировано:

- ✅ `test_5_4096` - работает
- ✅ `test_5_4_4096` - работает
- ✅ `generateSine` - работает
- ✅ `generate_from_json` - работает
- ✅ Thread safety DContext - работает
- ✅ Разные амплитуды - работает (энергия = амплитуда²)
- ✅ FFT цепочка - работает

---

## 💡 Для дальнейшей разработки:

1. **JSON парсер:** Можно подключить библиотеку JSON для чтения конфигов из файлов
2. **Разные типы сигналов:** Можно добавить XOR, модуляции, шум
3. **Разные сигналы на луч:** Архитектура готова для этого

---

**Вопросы?** Смотри тесты в `test_fft_chain.cpp` и `test_signal_generator.cpp`

---

*Создано Кодо для работы Алекса* 🎯
