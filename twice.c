#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
  {0, 0},
  {1, 2},
  {2, 4},
  {3, 6},
  {4, 8},
};

#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float(void)
{
  return ((float) rand() / (float) RAND_MAX);
}

float cost(float w)
{
  float result = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float x  = train[i][0];
    float y  = x * w;
    float d  = y - train[i][1];
    result += d*d;
  }
  result /= train_count;

  return result;
}

float dcost(float w)
{
  float result = 0.0f;
  size_t n = train_count;
  for (size_t i = 0; i < n; ++i) {
    float x  = train[i][0];
    float y  = train[i][1];
    result += 2*(x*w - y)*x;
  }
  result /= n;
  return result;
}

int main()
{
  // y = x*w + b;

  srand(time(0));
  float w = rand_float(); 

  // float eps = 1e-3;
  float rate = 1e-1;

  for (size_t i = 0; i < 1; ++i) {
    // float c = cost(w);
    float dw = dcost(w);// (cost(w + eps) - c)/eps;
    w -= rate * dw;
    printf("w: %f, cost: %f\n", w, cost(w));
  }
  printf("--------------------------------------\n");

  printf("w: %f, cost: %f\n", w, cost(w));

  for (size_t i = 0; i < 5; ++i) {
      printf("%zu -> %f\n", i, i*w);
  }

  return 0;
}
