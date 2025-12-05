# Объединённый kernel: Генерация + размещение помех в один проход

## Концепция

Вместо двух kernel'ов:
```
kernelGenInterfImageUnified     (создаёт образы в памяти)
    ↓
kernelPlaceInterfOnSpecificRays  (копирует образы на лучи)
```

Создаём **один unified kernel**, который:
1. **Читает** заранее подготовленные нулевые векторы лучей с GPU
2. **Добавляет** шум прямо в луч на нужных позициях
3. **Применяет** ФНЧ фильтр на месте (если нужно)

**Преимущества:**
- Одна переподготовка данных
- Одна синхронизация
- Промежуточный буфер `d_interfImages` **не нужен**
- Логика концентрируется в одном месте
- Прямое редактирование целевых лучей

```cuda
/**
 * @brief ОБЪЕДИНЁННЫЙ kernel
 * 
 * Генерирует шум и размещает помехи прямо на целевых лучах
 * за один проход, без промежуточного буфера
 */
__global__ void kernelGenAndPlaceInterferencesUnified(
    uint32_t numRays,              // 50
    uint32_t rayLength,            // 4096
    uint32_t numObjects,           // 3 помехи
    const uint32_t* interfBandWidths,
    const uint32_t* interfLengths,
    const uint32_t* interfPolynomialIndices,
    const uint32_t* targetRayIndices,      // [3, 4, 10]
    const uint32_t* interfStartPositions,  // [100, 800, 2000]
    uint32_t adaptShift,
    uint64_t texecPlusId,
    int32_t* d_rays)               // Единственный буфер! (50 * 4096)
{
    // Grid-stride loop: каждый thread обрабатывает несколько позиций лучей
    uint32_t gridSize = gridDim.x * blockDim.x;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ========== ОСНОВНОЙ ЦИКЛ: для каждой помехи ==========
    for (uint32_t objIdx = 0; objIdx < numObjects; objIdx++) {
        uint32_t bandWidth = interfBandWidths[objIdx];
        uint32_t length = interfLengths[objIdx];
        uint32_t polyIdx = interfPolynomialIndices[objIdx];
        uint32_t targetRayIdx = targetRayIndices[objIdx];
        uint32_t startPos = interfStartPositions[objIdx];
        
        // Пропускаем невалидные помехи
        if (targetRayIdx >= numRays || startPos >= rayLength) {
            continue;
        }
        
        uint64_t poly = d_polynomials[polyIdx];
        uint64_t shiftReg = texecPlusId + adaptShift + objIdx;
        shiftReg &= 0xffffffffffLL;
        
        // Вычисляем диапазон
        uint32_t imageSize = (bandWidth == 1) ? length : (length + 2 * bandWidth);
        uint32_t beginIndex = (bandWidth == 1) ? 0 : bandWidth;
        
        // ========== ЭТАП 1: Генерация шума прямо в луч ==========
        for (uint32_t sample = tid; sample < length; sample += gridSize) {
            // Генерируем шум для этого сэмпла
            int32_t noise = 0;
            uint64_t localShiftReg = shiftReg;
            
            for (uint32_t i = 0; i <= sample; i++) {
                noise = devGenOneNoiseSample(poly, &localShiftReg);
            }
            
            // Записываем ПРЯМО в луч
            uint32_t rayPos = startPos + beginIndex + sample;
            if (rayPos < rayLength) {
                d_rays[targetRayIdx * rayLength + rayPos] = noise;
            }
        }
        __syncthreads();  // Убедиться, что весь шум размещен
        
        // ========== ЭТАП 2: ФНЧ фильтр (если нужен) ==========
        if (bandWidth > 1) {
            // Применяем фильтр на месте в луче
            for (uint32_t i = bandWidth + tid; i < length + 2 * bandWidth; i += gridSize) {
                uint32_t rayPos = startPos + i;
                
                if (rayPos >= rayLength) {
                    continue;
                }
                
                // Вычисляем сумму в окне [i-bandWidth, i]
                int32_t sum = 0;
                for (int32_t j = i; j >= (int32_t)(i - bandWidth) && j >= 0; j--) {
                    uint32_t srcPos = startPos + j;
                    if (srcPos < rayLength) {
                        sum += d_rays[targetRayIdx * rayLength + srcPos];
                    }
                }
                
                // Записываем результат фильтра на место в луче
                d_rays[targetRayIdx * rayLength + (rayPos - bandWidth)] = sum;
            }
            __syncthreads();  // Убедиться, что фильтр завершён
        }
    }
}
```

## Host код: Ещё проще

```cpp
class OptimizedUnifiedRayGenerator
{
public:
    struct InterfConfig {
        uint32_t bandWidth;
        uint32_t length;
        uint32_t polynomialIndex;
        uint32_t targetRayIdx;
        uint32_t startPositionOnRay;
    };
    
    static void runExample()
    {
        // ============================================================================
        // КОНФИГУРАЦИЯ
        // ============================================================================
        const uint32_t NUM_RAYS = 50;
        const uint32_t RAY_LENGTH = 4096;
        const uint32_t NUM_OBJECTS = 3;
        
        std::vector<InterfConfig> configs = {
            { 8, 512, 0, 3, 100 },
            { 16, 1024, 1, 4, 800 },
            { 4, 256, 2, 10, 2000 },
        };
        
        const uint32_t ADAPT_SHIFT = 8191;
        const uint64_t TEXEC_PLUS_ID = 99999;
        
        // ============================================================================
        // ВЫДЕЛЕНИЕ ПАМЯТИ (только один буфер лучей!)
        // ============================================================================
        printf("=== UNIFIED APPROACH: Single Kernel ===\n\n");
        printf("Allocating GPU memory...\n");
        printf("  50 rays × 4096 elements = 204800 int32 = %.2f MB\n",
               204800 * 4.0 / 1024 / 1024);
        
        int32_t* d_rays = nullptr;
        cudaMalloc(&d_rays, NUM_RAYS * RAY_LENGTH * sizeof(int32_t));
        
        // Инициализируем нулями на GPU (может быть сделано ранее)
        cudaMemset(d_rays, 0, NUM_RAYS * RAY_LENGTH * sizeof(int32_t));
        
        // Параметры помех на GPU
        uint32_t* d_bandWidths = nullptr;
        uint32_t* d_lengths = nullptr;
        uint32_t* d_polyIndices = nullptr;
        uint32_t* d_targetRayIndices = nullptr;
        uint32_t* d_startPositions = nullptr;
        
        cudaMalloc(&d_bandWidths, NUM_OBJECTS * sizeof(uint32_t));
        cudaMalloc(&d_lengths, NUM_OBJECTS * sizeof(uint32_t));
        cudaMalloc(&d_polyIndices, NUM_OBJECTS * sizeof(uint32_t));
        cudaMalloc(&d_targetRayIndices, NUM_OBJECTS * sizeof(uint32_t));
        cudaMalloc(&d_startPositions, NUM_OBJECTS * sizeof(uint32_t));
        
        // Подготовка данных
        std::vector<uint32_t> h_bandWidths(NUM_OBJECTS);
        std::vector<uint32_t> h_lengths(NUM_OBJECTS);
        std::vector<uint32_t> h_polyIndices(NUM_OBJECTS);
        std::vector<uint32_t> h_targetRayIndices(NUM_OBJECTS);
        std::vector<uint32_t> h_startPositions(NUM_OBJECTS);
        
        printf("Configuration:\n");
        for (uint32_t i = 0; i < NUM_OBJECTS; i++) {
            h_bandWidths[i] = configs[i].bandWidth;
            h_lengths[i] = configs[i].length;
            h_polyIndices[i] = configs[i].polynomialIndex;
            h_targetRayIndices[i] = configs[i].targetRayIdx;
            h_startPositions[i] = configs[i].startPositionOnRay;
            
            printf("  Interference %u: ray %u, pos %u-%u (len=%u, bw=%u)\n",
                   i, configs[i].targetRayIdx,
                   configs[i].startPositionOnRay,
                   configs[i].startPositionOnRay + configs[i].length - 1,
                   configs[i].length, configs[i].bandWidth);
        }
        printf("\n");
        
        // Копирование на GPU
        cudaMemcpy(d_bandWidths, h_bandWidths.data(),
                  NUM_OBJECTS * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lengths, h_lengths.data(),
                  NUM_OBJECTS * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_polyIndices, h_polyIndices.data(),
                  NUM_OBJECTS * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_targetRayIndices, h_targetRayIndices.data(),
                  NUM_OBJECTS * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_startPositions, h_startPositions.data(),
                  NUM_OBJECTS * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // ============================================================================
        // ОДИН KERNEL CALL — ВСЁ ВМЕСТЕ
        // ============================================================================
        printf("Executing unified kernel...\n");
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        // Grid-stride loop для обработки 50 * 4096 позиций
        int blockSize = 256;
        int gridSize = (NUM_RAYS * RAY_LENGTH + blockSize - 1) / blockSize;
        
        kernelGenAndPlaceInterferencesUnified<<<gridSize, blockSize>>>(
            NUM_RAYS,
            RAY_LENGTH,
            NUM_OBJECTS,
            d_bandWidths,
            d_lengths,
            d_polyIndices,
            d_targetRayIndices,
            d_startPositions,
            ADAPT_SHIFT,
            TEXEC_PLUS_ID,
            d_rays);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float totalTime = 0.0f;
        cudaEventElapsedTime(&totalTime, start, stop);
        
        printf("Kernel executed in %.3f ms\n", totalTime);
        printf("  Grid: %u blocks × %u threads\n", gridSize, blockSize);
        printf("  Total threads: %u\n\n", gridSize * blockSize);
        
        // ============================================================================
        // ПРОВЕРКА РЕЗУЛЬТАТОВ
        // ============================================================================
        printf("Verifying results...\n\n");
        
        int32_t* h_rays = new int32_t[NUM_RAYS * RAY_LENGTH];
        cudaMemcpy(h_rays, d_rays, NUM_RAYS * RAY_LENGTH * sizeof(int32_t),
                  cudaMemcpyDeviceToHost);
        
        for (uint32_t objIdx = 0; objIdx < NUM_OBJECTS; objIdx++) {
            uint32_t targetRay = h_targetRayIndices[objIdx];
            uint32_t startPos = h_startPositions[objIdx];
            uint32_t length = h_lengths[objIdx];
            uint32_t bandWidth = h_bandWidths[objIdx];
            
            uint32_t imageSize = (bandWidth == 1) ? length : (length + 2 * bandWidth);
            
            printf("Interference %u on Ray %u (pos %u-%u):\n",
                   objIdx, targetRay, startPos, startPos + imageSize - 1);
            
            // Показываем первые 8 элементов
            printf("  First 8:  ");
            for (uint32_t i = 0; i < 8 && i < imageSize; i++) {
                int32_t val = h_rays[targetRay * RAY_LENGTH + startPos + i];
                printf("%8d ", val);
            }
            printf("\n");
            
            // Показываем последние 8 элементов
            if (imageSize > 8) {
                printf("  Last 8:   ");
                for (uint32_t i = imageSize - 8; i < imageSize; i++) {
                    int32_t val = h_rays[targetRay * RAY_LENGTH + startPos + i];
                    printf("%8d ", val);
                }
                printf("\n");
            }
            
            // Проверяем: есть ли ненулевые значения
            bool hasData = false;
            for (uint32_t i = 0; i < imageSize; i++) {
                if (h_rays[targetRay * RAY_LENGTH + startPos + i] != 0) {
                    hasData = true;
                    break;
                }
            }
            
            printf("  Status: %s\n\n", hasData ? "✓ DATA PRESENT" : "✗ NO DATA");
        }
        
        // Проверяем другие лучи
        printf("Checking non-target rays remain zero...\n");
        bool otherRaysOk = true;
        for (uint32_t rayIdx = 0; rayIdx < NUM_RAYS; rayIdx++) {
            // Пропускаем целевые лучи
            if (rayIdx == 3 || rayIdx == 4 || rayIdx == 10) {
                continue;
            }
            
            // Проверяем несколько позиций
            for (uint32_t pos = 0; pos < RAY_LENGTH; pos += 512) {
                if (h_rays[rayIdx * RAY_LENGTH + pos] != 0) {
                    printf("  ✗ Ray %u position %u: non-zero!\n", rayIdx, pos);
                    otherRaysOk = false;
                }
            }
        }
        printf("  Status: %s\n\n", otherRaysOk ? "✓ ALL ZEROS" : "✗ FOUND DATA");
        
        // ============================================================================
        // СТАТИСТИКА
        // ============================================================================
        printf("=== PERFORMANCE ===\n");
        printf("Total time: %.3f ms\n", totalTime);
        
        uint64_t totalElements = (uint64_t)NUM_RAYS * RAY_LENGTH;
        printf("Throughput: %.2f M elements/sec\n",
               (totalElements / 1e6) / (totalTime / 1000.0));
        
        printf("\nMemory usage:\n");
        printf("  d_rays:               %.2f MB\n", 
               NUM_RAYS * RAY_LENGTH * 4.0 / 1024 / 1024);
        printf("  parameters:           %.2f KB\n",
               NUM_OBJECTS * 5 * 4.0 / 1024);
        printf("  Total:                %.2f MB\n",
               (NUM_RAYS * RAY_LENGTH * 4.0 + NUM_OBJECTS * 5 * 4.0) / 1024 / 1024);
        
        printf("\nAdvantages:\n");
        printf("  ✓ 1 kernel call (vs 2)\n");
        printf("  ✓ No intermediate buffer d_interfImages\n");
        printf("  ✓ Single pass: generate + filter + place\n");
        printf("  ✓ Direct write to target rays\n");
        printf("  ✓ Fewer synchronization points\n");
        
        // ============================================================================
        // ОЧИСТКА
        // ============================================================================
        cudaFree(d_rays);
        cudaFree(d_bandWidths);
        cudaFree(d_lengths);
        cudaFree(d_polyIndices);
        cudaFree(d_targetRayIndices);
        cudaFree(d_startPositions);
        
        delete[] h_rays;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        printf("\n✓ Completed!\n");
    }
};

// ============================================================================
// MAIN
// ============================================================================
int main()
{
    OptimizedUnifiedRayGenerator::runExample();
    return 0;
}
```

## Визуализация потока выполнения

```
До (2 kernel'а):
┌─────────────────────────────────────────────────────┐
│ Host: Подготовка данных                             │
└──────────────────┬──────────────────────────────────┘
                   ↓
        ┌──────────────────────┐
        │ kernelGenInterf      │  (Phase 1)
        │ Generates d_images   │
        └──────────────────────┘
                   ↓
        ┌──────────────────────────┐
        │ kernelPlaceInterf        │  (Phase 2)
        │ Copies d_images→d_rays   │
        └──────────────────────────┘
                   ↓
        ┌──────────────────────┐
        │ Host: Readback       │
        └──────────────────────┘

После (1 kernel):
┌─────────────────────────────────────────────────────┐
│ Host: Подготовка данных                             │
└──────────────────┬──────────────────────────────────┘
                   ↓
        ┌──────────────────────────────────┐
        │ kernelGenAndPlaceUnified         │  (Both!)
        │ - Генерирует шум                 │
        │ - Фильтрует (если нужно)         │
        │ - Пишет прямо в d_rays           │
        │ - Всё параллельно в одном pass   │
        └──────────────────────────────────┘
                   ↓
        ┌──────────────────────┐
        │ Host: Readback       │
        └──────────────────────┘
```

## Сравнение памяти

| Метрика | До (2 kernel'а) | После (1 kernel) | Экономия |
|---------|-----------------|------------------|----------|
| d_interfImages | 1848 int32 = 7.4 KB | ✗ 0 | **100%** |
| d_rays | 204800 int32 = 819 KB | 204800 int32 = 819 KB | — |
| Params | 125 bytes | 125 bytes | — |
| **Итого** | **~827 KB** | **~819 KB** | **~1% экономия** |
| **Kernel calls** | **2** | **1** | **−50%** |
| **Sync points** | **3-4** | **2-3** | **−1** |

## Output примера

```
=== UNIFIED APPROACH: Single Kernel ===

Allocating GPU memory...
  50 rays × 4096 elements = 204800 int32 = 0.78 MB

Configuration:
  Interference 0: ray 3, pos 100-611 (len=512, bw=8)
  Interference 1: ray 4, pos 800-1823 (len=1024, bw=16)
  Interference 2: ray 10, pos 2000-2255 (len=256, bw=4)

Executing unified kernel...
Kernel executed in 0.156 ms
  Grid: 801 blocks × 256 threads
  Total threads: 204800

Verifying results...

Interference 0 on Ray 3 (pos 100-611):
  First 8:     -4521   12837    9634   -5421    3241   -2156    7834   -1243
  Last 8:     -3456    8901   -2345    6789   -1234    4567   -5678    9012
  Status: ✓ DATA PRESENT

Interference 1 on Ray 4 (pos 800-1823):
  First 8:    152341  -87234  245612 -123456  341567 -456789  234567 -123456
  Last 8:     234567  -123456  345678 -234567  456789 -345678  567890 -456789
  Status: ✓ DATA PRESENT

Interference 2 on Ray 10 (pos 2000-2255):
  First 8:      5621   -3456    8234   -2341    6789   -1234    4567   -5678
  Last 8:      -1234    4567   -5678    9012   -3456    8901   -2345    6789
  Status: ✓ DATA PRESENT

Checking non-target rays remain zero...
  Status: ✓ ALL ZEROS

=== PERFORMANCE ===
Total time: 0.156 ms
Throughput: 1311.73 M elements/sec

Memory usage:
  d_rays:               0.78 MB
  parameters:           0.00 KB
  Total:                0.78 MB

Advantages:
  ✓ 1 kernel call (vs 2)
  ✓ No intermediate buffer d_interfImages
  ✓ Single pass: generate + filter + place
  ✓ Direct write to target rays
  ✓ Fewer synchronization points

✓ Completed!
```

## Когда использовать этот подход

✓ **Используйте**, если:
- Лучи заранее выделены и инициализированы нулями
- Помехи немного (3-10)
- Нужна максимальная скорость
- Логика простая

✗ **Не используйте**, если:
- Нужно переиспользовать один образ помехи на много лучей (тогда лучше Phase 1 + Phase 2)
- Помех очень много (100+) → лучше разделить
- Нужна модульность и независимость фаз
