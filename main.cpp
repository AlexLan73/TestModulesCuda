#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include "src/RandomGeneratorGPU.h"


int main()
{
  std::cout << "Test lib CUDA main" << std::endl;

  unsigned long long seed = 12345;
  size_t num_points = 2500000;

  RandomGeneratorGPU generator(seed, 100.0f, num_points);
  generator.generate();
  std::cout << "Sum before filter: " << generator.getSum() << std::endl;
  generator.applyLowPassConvolution(10);
  std::cout << "Sum after filter: " << generator.getSum() << std::endl;
  generator.printFirst10();

  return 0;
}

