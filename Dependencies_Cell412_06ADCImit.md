# Зависимости класса ModelNS23817A620S1823::Cell412_06ADCImit

Документ содержит полный перечень файлов, классов и структур, необходимых для переноса класса `Cell412_06ADCImit` из проекта `/home/alex/C++/model-dsp` в проект TestModulesCuda.

**Дата создания:** 2025-12-05

---

## 1. Описание класса

**Полное имя:** `ModelNS23817A620S1823::Cell412_06ADCImit`

**Назначение:** Класс имитатора выхода АЦП ячейки расчёта каналов 412-06.

**Основные методы:**
- `Cell412_06ADCImit()` - конструктор
- `~Cell412_06ADCImit()` - деструктор
- `getChannelData(uint32_t chanNum, int32_t channelData[])` - получение данных канала
- `setControlMessage(const ControlMsg& controlMsg)` - установка управляющего сообщения

**Внутренние методы генерации:**
- `genSelfNoise()` - генерация собственного шума
- `genInterfNoise()` - генерация шума помехи
- `genInterfImage()` - генерация образа помехи
- `genSignalImage()` - генерация образа сигнала
- `genObject()` - генерация объекта
- `genCarrierFreq()` - генерация несущей частоты
- `calcNk()`, `calcNdk()` - вспомогательные расчёты
- `applyKaiserOnSignal()` - наложение фильтра Кайзера

---

## 2. Основные файлы класса

| Файл | Путь относительно model-dsp | Строки кода |
|------|---------------------------|-------------|
| c412_06ADCImitator.h | `src/c412_06ADCImitator.h` | ~104 |
| c412_06ADCImitator.cpp | `src/c412_06ADCImitator.cpp` | ~407 |

---

## 3. Дерево зависимостей

```
Cell412_06ADCImit
├── controlMsg.h (ControlMsg class)
│   ├── sharedConst.h
│   ├── controlMsgModelTypes.h
│   │   ├── sharedConst.h
│   │   ├── xyChanCoords.h
│   │   ├── arCoefsArray.h
│   │   ├── dfGroups.h
│   │   ├── trpr.h
│   │   │   └── libvork/e2trpr.h
│   │   ├── ippa.h
│   │   │   └── libvork/e2ippa.h
│   │   ├── trimts.h
│   │   │   └── model/e2trimts.h
│   │   │       └── libvork/e2trims.h
│   │   └── model/modelInterface.h
│   ├── trkp.h
│   │   └── model/e2trkp.h
│   │       └── libvork/e2trkp.h
│   ├── callbacks.h
│   ├── model/controlMsgSharedTypes.h
│   │   └── model/modelSharedConst.h
│   │       └── model/minDefs.h
│   └── model/modelInterface.h
├── helpers.h
│   ├── model/controlMsgSharedTypes.h
│   ├── sliceFile.h
│   └── controlMsgModelTypes.h
└── sharedConst.h
    └── model/modelSharedConst.h
```

---

## 4. Полный список файлов для копирования

### 4.1. Исходный код (src/)

| # | Файл | Описание |
|---|------|----------|
| 1 | `src/c412_06ADCImitator.h` | **Главный класс** |
| 2 | `src/c412_06ADCImitator.cpp` | **Реализация главного класса** |
| 3 | `src/controlMsg.h` | Класс управляющего сообщения |
| 4 | `src/controlMsg.cpp` | Реализация ControlMsg |
| 5 | `src/helpers.h` | Вспомогательные функции |
| 6 | `src/helpers.cpp` | Реализация helpers |
| 7 | `src/sharedConst.h` | Константы SharedConst |
| 8 | `src/controlMsgModelTypes.h` | Типы данных CMsgModelTypes |
| 9 | `src/trkp.h` | Класс Trkp (калибровки/АФР) |
| 10 | `src/trkp.cpp` | Реализация Trkp |
| 11 | `src/callbacks.h` | Структуры коллбеков |
| 12 | `src/xyChanCoords.h` | Класс координат каналов |
| 13 | `src/xyChanCoords.cpp` | Реализация XYChanCoords |
| 14 | `src/arCoefsArray.h` | Класс амплитудного распределения |
| 15 | `src/arCoefsArray.cpp` | Реализация ArCoefsArray |
| 16 | `src/dfGroups.h` | Класс пеленгационных групп |
| 17 | `src/dfGroups.cpp` | Реализация DFGroups |
| 18 | `src/trpr.h` | Класс тактового расписания приёмника |
| 19 | `src/trpr.cpp` | Реализация Trpr |
| 20 | `src/ippa.h` | Класс активных помехопостановщиков |
| 21 | `src/ippa.cpp` | Реализация Ippa |
| 22 | `src/trimts.h` | Класс тактового расписания имитатора |
| 23 | `src/trimts.cpp` | Реализация Trimts |
| 24 | `src/sliceFile.h` | Класс файлов срезов |
| 25 | `src/sliceFile.cpp` | Реализация SliceFile |
| 26 | `src/wlbArray.h` | Класс массива WLB (используется в ControlMsg) |
| 27 | `src/wlbArray.cpp` | Реализация WlbArray |

### 4.2. Заголовочные файлы модели (include/model/)

| # | Файл | Описание |
|---|------|----------|
| 28 | `include/model/controlMsgSharedTypes.h` | Типы CMsgSharedTypes (ImitStrobe, ImitObject, ImitTact) |
| 29 | `include/model/modelSharedConst.h` | Общие константы SharedConst |
| 30 | `include/model/minDefs.h` | Базовые определения (Vector2d, Vector3d, Polarized) |
| 31 | `include/model/modelInterface.h` | Интерфейс модели |
| 32 | `include/model/e2trkp.h` | Структура E2Trkp |
| 33 | `include/model/e2trimts.h` | Определения для ТРИМЦ |
| 34 | `include/model/str_uco.h` | Буфер УЦОПО |

### 4.3. Библиотека libvork (lib/libvork/include/libvork/)

| # | Файл | Описание |
|---|------|----------|
| 35 | `e2ippa.h` | Структура E2IPPA (активные помехи) |
| 36 | `e2trpr.h` | Структура E2TRPR (тактовое расписание приёмника) |
| 37 | `e2trkp.h` | Структуры EUTRKP (калибровки) |
| 38 | `e2trims.h` | Структура TrimtsMsg (тактовое расписание имитатора) |
| 39 | `parsig.h` | Структура PAR_SIGN (параметры сигнала) |
| 40 | `ucopo.h` | Структуры УЦОПО |
| 41 | `defs.h` | Общие определения libvork |
| 42 | `rtnetxx.h` | RTNet типы |
| 43 | `libmpk.h` | Типы МПК |

### 4.4. Библиотека libutils (lib/libutils/include/libutils/)

| # | Файл | Описание |
|---|------|----------|
| 44 | `sumtype.h` | SumType (вариантный тип) |
| 45 | `result.h` | Result<T, E> (тип результата с ошибкой) |
| 46 | `alen.h` | Утилиты массивов |
| 47 | `boconv.h` | Преобразование порядка байт |

---

## 5. Ключевые структуры данных

### 5.1. Namespace ModelNS23817A620S1823

#### Классы:
- `Cell412_06ADCImit` - главный класс
- `ControlMsg` - управляющее сообщение
- `Trkp` - калибровки/АФР
- `Trpr` - тактовое расписание приёмника
- `Trimts` - тактовое расписание имитатора
- `Ippa` - активные помехопостановщики
- `XYChanCoords` - координаты каналов
- `ArCoefsArray` - амплитудное распределение
- `DFGroups` - пеленгационные группы
- `WlbArray` - массив коэффициентов ЦФНЧ
- `SliceFile` - файлы срезов

#### Структуры (CMsgSharedTypes):
- `ImitStrobe` - параметры имитируемого строба
- `ImitObject` - параметры объекта (помеха/сигнал)
- `ImitTact` - параметры имитируемого такта
- `ChanCoords` - координаты канала

#### Структуры (CMsgModelTypes):
- `StationParams` - параметры станции
- `StationConfig` - конфигурация станции
- `InputMessages` - входные сообщения

### 5.2. Namespace SharedConst

#### Перечисления:
- `StationType` - тип станции (voronezhM, voronezhSM, mrik)
- `PolarizationTypes` - тип поляризации (vertical, horizontal, both)
- `ObjectTypes` - тип объекта (signal, interference)
- `PolarsCombining` - способ объединения поляризаций
- `PhasingType` - тип фазирования

#### Константы:
- `maxChannelsInCellSM = 15`
- `maxChannelsInCellM = 4`
- `maxChannelsInCellMrik = 8`
- `maxInterfBandWidth = 1024`
- `maxRays = 16`
- `maxDFGroups = 10`
- и другие

---

## 6. Порядок подключения заголовков

```cpp
// 1. Стандартные библиотеки
#include <stdint.h>
#include <cmath>
#include <cstring>
#include <complex>
#include <vector>
#include <memory>
#include <iostream>

// 2. Библиотека libutils
#include <libutils/sumtype.h>
#include <libutils/result.h>

// 3. Библиотека libvork
#include <libvork/e2ippa.h>
#include <libvork/e2trpr.h>
#include <libvork/e2trkp.h>
#include <libvork/e2trims.h>
#include <libvork/parsig.h>

// 4. Заголовки модели
#include <model/minDefs.h>
#include <model/modelSharedConst.h>
#include <model/controlMsgSharedTypes.h>
#include <model/e2trkp.h>
#include <model/e2trimts.h>
#include <model/modelInterface.h>

// 5. Исходники src/
#include "sharedConst.h"
#include "callbacks.h"
#include "sliceFile.h"
#include "xyChanCoords.h"
#include "arCoefsArray.h"
#include "dfGroups.h"
#include "ippa.h"
#include "trimts.h"
#include "trpr.h"
#include "trkp.h"
#include "controlMsgModelTypes.h"
#include "helpers.h"
#include "controlMsg.h"
#include "c412_06ADCImitator.h"
```

---

## 7. Пример минимального использования

```cpp
#include "c412_06ADCImitator.h"
#include "controlMsg.h"

int main() {
    using namespace ModelNS23817A620S1823;
    
    // Создание имитатора
    Cell412_06ADCImit adcImit;
    
    // Создание управляющего сообщения (требует конфигурации)
    // ControlMsg controlMsg;
    
    // Установка параметров
    // adcImit.setControlMessage(controlMsg);
    
    // Получение данных канала
    // int32_t channelData[...];
    // adcImit.getChannelData(chanNum, channelData);
    
    return 0;
}
```

---

## 8. Предупреждения и замечания

### 8.1. Сложность переноса
- **Высокая**: класс `ControlMsg` очень сложный (~600 строк), имеет много зависимостей
- Требуются две внешние библиотеки: `libvork` и `libutils`
- Потребуется адаптация include paths

### 8.2. Необходимые изменения
1. Создать структуру каталогов:
   ```
   TestModulesCuda/
   ├── model-dsp/
   │   ├── src/
   │   ├── include/model/
   │   └── lib/
   │       ├── libvork/include/libvork/
   │       └── libutils/include/libutils/
   ```

2. Обновить CMakeLists.txt для включения новых путей

3. Возможно понадобится создать stub-реализации для функций libvork/libutils

### 8.3. Альтернативный подход
Если нужен только класс `Cell412_06ADCImit` без полной функциональности `ControlMsg`, можно:
1. Извлечь только необходимые структуры данных
2. Создать упрощённую версию управляющего сообщения
3. Убрать зависимости от сетевых структур (E2TRPR, E2IPPA, etc.)

---

## 9. Ссылки

- **Документация Doxygen:** `/home/alex/C++/model-dsp/doxydoc/html/classModelNS23817A620S1823_1_1Cell412__06ADCImit.html`
- **Исходный проект:** `/home/alex/C++/model-dsp`
- **Целевой проект:** `/home/alex/C++/TestModulesCuda`

---

*Документ сформирован автоматически на основе анализа исходного кода проекта model-dsp.*

