# DataContext library CMakeLists.txt

add_library(DataContext STATIC
    core/DContext.cpp
    core/Validator.cpp
    parser/convert_bin.cpp
)

# PRIVATE — нужны только для сборки DataContext (например, nlohmann/json)
target_include_directories(DataContext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann
)

# Линкуем Interface как PUBLIC зависимость
target_link_libraries(DataContext PUBLIC Interface)

target_include_directories(DataContext PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/parser
    ${CMAKE_SOURCE_DIR}/   # наверное лишее
)
