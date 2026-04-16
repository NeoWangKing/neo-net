// xor.c
#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

float td[] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 0,
};

int main(void)
{
  srand(time(0));
  // srand(69);

  size_t arch[] = {2, 2, 1};
  size_t arch_count = ARRAY_LEN(arch);
  size_t input_num = arch[0];
  size_t output_num = arch[arch_count - 1];
  size_t stride = input_num + output_num;
  size_t n = sizeof(td)/sizeof(td[0])/stride;

  Mat ti = { .rows = n, .cols = input_num, .stride = stride, .es = td };
  Mat to = { .rows = n, .cols = output_num, .stride = stride, .es = td + input_num };

  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN g  = nn_alloc(arch, ARRAY_LEN(arch));
  nn_rand(nn, 0, 1);

  float rate = 1;

  printf("cost = %f\n", nn_cost(nn, ti, to));
  for (size_t i = 0; i < 10*1000; ++i) {
    nn_backprop(nn, g, ti, to);
    nn_learn(nn, g, rate);
    printf("%zu: cost = %f\n", i + 1, nn_cost(nn, ti, to));
  }
  printf("cost = %f\n", nn_cost(nn, ti, to));

#if 1
  printf("----------------------------------\n");
  NN_PRINT(nn);  
  printf("----------------------------------\n");
  
  size_t total_combinations = 1 << input_num;  // 2^4 = 16
  
  for (size_t combo = 0; combo < total_combinations; ++combo) {
    printf("INPUT: [");
    for (size_t k = 0; k < input_num; ++k) {
      float value = (combo >> k) & 1 ? 1.0f : 0.0f;
      MAT_AT(NN_INPUT(nn), 0, k) = value;
      printf("%.0f", value);
      if (k < input_num - 1) printf(", ");
    }
    printf("]");
    
    nn_forward(nn);
    printf(" -> ");
    
    printf("OUTPUT: [");
    for (size_t k = 0; k < output_num; ++k) {
      printf("%.4f", MAT_AT(NN_OUTPUT(nn), 0, k));
      if (k < output_num - 1) printf(", ");
    }
    printf("]\n");
  }
#endif

  return 0;
}
