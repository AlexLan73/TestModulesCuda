# Оптимизированный объединённый kernel для генерации образа помехи

## Основная идея

Вместо двух отдельных kernel'ов:
```
kernelGenInterfImage (Phase 1: шум + инициализация)
  ↓ __syncthreads()
kernelGenInterfImage (Phase 2: ФНЧ фильтр)
```

Создаём **один unified kernel** с условиями по признаку `bandWidth`:

```cuda
__global__ void kernelGenInterfImageUnified(
    uint32_t numObjects,
    const uint32_t* interfBandWidths,
    const uint32_t* interfLengths,
    const uint32_t* interfPolynomialIndices,
    uint32_t adaptShift,
    uint64_t texecPlusId,
    int32_t* d_interfImages,
    const uint32_t* d_interfImageOffsets,
    const uint32_t* d_interfImageSizes)
{
    uint32_t objIdx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    
    if (objIdx >= numObjects) {
        return;
    }
    
    uint32_t bandWidth = interfBandWidths[objIdx];
    uint32_t length = interfLengths[objIdx];
    uint32_t polyIdx = interfPolynomialIndices[objIdx];
    uint32_t imageOffset = d_interfImageOffsets[objIdx];
    uint32_t imageSize = d_interfImageSizes[objIdx];
    
    int32_t* objectImage = &d_interfImages[imageOffset];
    uint64_t poly = d_polynomials[polyIdx];
    
    // Инициализация шифт-регистра
    uint64_t shiftReg = texecPlusId + adaptShift;
    shiftReg &= 0xffffffffffLL;
    
    // ========== ЭТАП 1: Инициализация и генерация шума ==========
    for (uint32_t i = tid; i < imageSize; i += blockDim.x) {
        objectImage[i] = 0;  // Инициализация нулями
    }
    __syncthreads();
    
    uint32_t beginIndex = (bandWidth == 1) ? 0 : bandWidth;
    
    // Генерируем шум
    for (uint32_t sample = tid; sample < length; sample += blockDim.x) {
        int32_t noise = 0;
        uint64_t localShiftReg = shiftReg;
        
        for (uint32_t i = 0; i <= sample; i++) {
            noise = devGenOneNoiseSample(poly, &localShiftReg);
        }
        
        objectImage[beginIndex + sample] = noise;
    }
    __syncthreads();
    
    // ========== ЭТАП 2: ФНЧ фильтр (если bandWidth > 1) ==========
    // УСЛОВИЕ ПО ПРИЗНАКУ: проверяем bandWidth
    if (bandWidth > 1) {
        // Применяем фильтр только если нужно
        for (uint32_t i = bandWidth + tid; i < length + 2 * bandWidth; i += blockDim.x) {
            int32_t sum = 0;
            for (int32_t j = i; j >= (int32_t)(i - bandWidth) && j >= 0; j--) {
                sum += objectImage[j];
            }
            objectImage[i - bandWidth] = sum;
        }
        __syncthreads();
    }
    // Если bandWidth == 1, пропускаем этап 2 целиком
}
```

## Вариант 2: С минимальными синхронизациями

Если шум и фильтр работают на **разных диапазонах памяти**, можем избежать второго `__syncthreads()`:

```cuda
__global__ void kernelGenInterfImageOptimized(
    uint32_t numObjects,
    const uint32_t* interfBandWidths,
    const uint32_t* interfLengths,
    const uint32_t* interfPolynomialIndices,
    uint32_t adaptShift,
    uint64_t texecPlusId,
    int32_t* d_interfImages,
    const uint32_t* d_interfImageOffsets,
    const uint32_t* d_interfImageSizes)
{
    uint32_t objIdx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    
    if (objIdx >= numObjects) {
        return;
    }
    
    uint32_t bandWidth = interfBandWidths[objIdx];
    uint32_t length = interfLengths[objIdx];
    uint32_t polyIdx = interfPolynomialIndices[objIdx];
    uint32_t imageOffset = d_interfImageOffsets[objIdx];
    uint32_t imageSize = d_interfImageSizes[objIdx];
    
    int32_t* objectImage = &d_interfImages[imageOffset];
    uint64_t poly = d_polynomials[polyIdx];
    
    uint64_t shiftReg = texecPlusId + adaptShift;
    shiftReg &= 0xffffffffffLL;
    
    // ========== ФАЗА 1: Инициализация + Генерация ==========
    for (uint32_t i = tid; i < imageSize; i += blockDim.x) {
        objectImage[i] = 0;
    }
    __syncthreads();  // Барьер 1: Убедиться, что всё инициализировано
    
    uint32_t beginIndex = (bandWidth == 1) ? 0 : bandWidth;
    
    // Генерируем шум в диапазон [beginIndex, beginIndex + length)
    for (uint32_t sample = tid; sample < length; sample += blockDim.x) {
        int32_t noise = 0;
        uint64_t localShiftReg = shiftReg;
        
        for (uint32_t i = 0; i <= sample; i++) {
            noise = devGenOneNoiseSample(poly, &localShiftReg);
        }
        
        objectImage[beginIndex + sample] = noise;
    }
    
    // ========== ФАЗА 2: ФНЧ (условно) ==========
    // Если bandWidth > 1, применяем фильтр
    if (bandWidth > 1) {
        __syncthreads();  // Барьер 2: Убедиться, что весь шум готов
        
        // Фильтр работает на [0, length + 2*bandWidth)
        // Читает из уже заполненных позиций
        for (uint32_t i = bandWidth + tid; i < length + 2 * bandWidth; i += blockDim.x) {
            int32_t sum = 0;
            for (int32_t j = i; j >= (int32_t)(i - bandWidth) && j >= 0; j--) {
                sum += objectImage[j];
            }
            objectImage[i - bandWidth] = sum;
        }
    }
    // Если bandWidth == 1: нет второго __syncthreads() и нет фильтра
}
```

## Вариант 3: С warp-level операциями (для bandWidth < 32)

Для малых значений `bandWidth` можем использовать **warp shuffle**:

```cuda
__device__ __forceinline__ int32_t warpReduceSum(int32_t val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ void kernelGenInterfImageWarpOptimized(
    uint32_t numObjects,
    const uint32_t* interfBandWidths,
    const uint32_t* interfLengths,
    const uint32_t* interfPolynomialIndices,
    uint32_t adaptShift,
    uint64_t texecPlusId,
    int32_t* d_interfImages,
    const uint32_t* d_interfImageOffsets,
    const uint32_t* d_interfImageSizes)
{
    uint32_t objIdx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid / 32;  // warp id
    uint32_t lane = tid % 32; // lane in warp
    
    if (objIdx >= numObjects) {
        return;
    }
    
    uint32_t bandWidth = interfBandWidths[objIdx];
    uint32_t length = interfLengths[objIdx];
    uint32_t polyIdx = interfPolynomialIndices[objIdx];
    uint32_t imageOffset = d_interfImageOffsets[objIdx];
    uint32_t imageSize = d_interfImageSizes[objIdx];
    
    int32_t* objectImage = &d_interfImages[imageOffset];
    uint64_t poly = d_polynomials[polyIdx];
    
    uint64_t shiftReg = texecPlusId + adaptShift;
    shiftReg &= 0xffffffffffLL;
    
    // Инициализация
    for (uint32_t i = tid; i < imageSize; i += blockDim.x) {
        objectImage[i] = 0;
    }
    __syncthreads();
    
    uint32_t beginIndex = (bandWidth == 1) ? 0 : bandWidth;
    
    // Генерация шума
    for (uint32_t sample = tid; sample < length; sample += blockDim.x) {
        int32_t noise = 0;
        uint64_t localShiftReg = shiftReg;
        
        for (uint32_t i = 0; i <= sample; i++) {
            noise = devGenOneNoiseSample(poly, &localShiftReg);
        }
        
        objectImage[beginIndex + sample] = noise;
    }
    __syncthreads();
    
    // ФНЧ фильтр с warp optimization
    if (bandWidth > 1 && bandWidth <= 32) {
        for (uint32_t i = tid; i < length + bandWidth; i += blockDim.x) {
            int32_t sum = 0;
            
            // Каждый lane в warp читает один элемент окна
            if (lane < bandWidth) {
                int32_t idx = i + lane;
                if (idx < length + 2 * bandWidth) {
                    sum = objectImage[idx];
                }
            }
            
            // Суммируем через warp shuffle
            sum = warpReduceSum(sum);
            
            if (lane == 0) {
                objectImage[i - bandWidth] = sum;
            }
        }
    } else if (bandWidth > 1) {
        // Fallback для больших bandWidth
        for (uint32_t i = bandWidth + tid; i < length + 2 * bandWidth; i += blockDim.x) {
            int32_t sum = 0;
            for (int32_t j = i; j >= (int32_t)(i - bandWidth) && j >= 0; j--) {
                sum += objectImage[j];
            }
            objectImage[i - bandWidth] = sum;
        }
    }
}
```

## Интеграция в host код

```cpp
class CudaInterfImageGeneratorOptimized
{
public:
    /**
     * @brief Unified генерация образов помех с одним kernel call
     */
    static void genInterfImagesUnified(
        uint32_t numObjects,
        const InterfConfig* h_configs,
        uint32_t adaptShift,
        uint64_t texecPlusId,
        uint32_t* d_bandWidths,
        uint32_t* d_lengths,
        uint32_t* d_polyIndices,
        uint32_t* d_imageOffsets,
        uint32_t* d_imageSizes,
        int32_t* d_interfImages,
        uint32_t totalImageSize)
    {
        // Один kernel call вместо двух!
        int blockSize = 128;
        int gridSize = numObjects;
        
        kernelGenInterfImageUnified<<<gridSize, blockSize>>>(
            numObjects,
            d_bandWidths,
            d_lengths,
            d_polyIndices,
            adaptShift,
            texecPlusId,
            d_interfImages,
            d_imageOffsets,
            d_imageSizes);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "kernelGenInterfImageUnified error: %s\n",
                    cudaGetErrorString(err));
        }
    }
};
```

## Преимущества объединения

| Аспект | Было (2 kernel) | Стало (1 kernel) |
|--------|-----------------|-----------------|
| **Kernel calls** | 2 × (setup + launch) | 1 × (setup + launch) |
| **Host-Device overhead** | × 2 | × 1 |
| **GPU Launch latency** | ~µs × 2 | ~µs × 1 |
| **Synchronization points** | 2 `__syncthreads()` | 1-2 (условно) |
| **Условие по bandWidth** | Неявное | **Явное в признаке** |

### Когда использовать каждый вариант:

1. **Вариант 1 (базовый unified)** — универсален, не требует изменений в памяти
2. **Вариант 2 (оптимизированный)** — если bandWidth ≠ 1 часто, экономим на `__syncthreads()`
3. **Вариант 3 (warp-optimized)** — если bandWidth < 32, используем warp shuffle для ускорения фильтра

## Теоретическое ускорение

Если предположить:
- Kernel launch overhead: ~1-5 µs
- numObjects = 4

**Экономия на вызовах:**
```
Было:      4 objects × 2 kernels × ~2 µs = 16 µs
Стало:     4 objects × 1 kernel  × ~2 µs = 8 µs overhead
Экономия:  ~50% на launch overhead для фазы 1-2
```

Plus экономия на **host-side GPU queue management**, синхронизациях и memory barriers.

## Размещение помех на лучах (Phase 3) - остаётся без изменений

```cuda
// kernelPlaceInterfOnRays — остаётся как есть
// Это отдельная операция с другой логикой доступа к памяти
__global__ void kernelPlaceInterfOnRays_Simple(
    uint32_t numRays,
    uint32_t rayLength,
    uint32_t numObjects,
    const uint32_t* interfStartPositions,
    const uint32_t* interfImageSizes,
    const int32_t* d_interfImages,
    const uint32_t* d_interfImageOffsets,
    int32_t* d_rays)
{
    // Без изменений — работает после объединённого kernelGenInterfImageUnified
}
```

## Полный пример вызова

```cpp
// Было (2 separate kernels + phase 2)
genInterfImages(...);  // 1 kernel call
placeInterfOnRays(...); // 1 kernel call
// Итого: 2 kernel calls + synchronization

// Стало (unified kernel)
genInterfImagesUnified(...);  // 1 unified kernel call
placeInterfOnRays(...);       // 1 kernel call
// Итого: 2 kernel calls, но меньше overhead
```

Какой вариант вам подходит больше? Могу доработать конкретный код!
