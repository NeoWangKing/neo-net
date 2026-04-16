#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float sigmoidf(float x)
{
  return 1.f / (1.f + expf(-x));
}

typedef float sample[3];

// OR-gate
sample train_or[] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 1},
};

// AND-gate
sample train_and[] = { 
  {0, 0, 0},
  {0, 1, 0},
  {1, 0, 0},
  {1, 1, 1},
};

// NAND-gate
sample train_nand[] = { 
  {0, 0, 1},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0},
};

// XOR-gate
sample train_xor[] = { 
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0},
};

sample *train = train_or;
size_t train_count = 4;
// #define train_count (sizeof(train)/sizeof(train[0]))

float rand_float(void)
{
  return ((float) rand() / (float) RAND_MAX);
}

float cost(float w1, float w2, float b)
{
  float result = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y  = sigmoidf(x1 * w1 + x2 * w2 + b);
    float d  = y - train[i][2];
    result += d*d;
  }
  result /= train_count;

  return result;
}

void dcost(float eps,
           float w1, float w2, float b,
           float *dw1, float *dw2, float *db)
{
  float c = cost(w1, w2, b);
  *dw1 = (cost(w1 + eps, w2, b) - c)/eps;
  *dw2 = (cost(w1, w2 + eps, b) - c)/eps;
  *db  = (cost(w1, w2, b + eps) - c)/eps;
}

void gcost(float w1, float w2, float b,
           float *dw1, float *dw2, float *db)
{
  *dw1 = 0;
  *dw2 = 0;
  *db  = 0;

  size_t n = train_count;
  for (size_t i = 0; i < n; ++i) {
    float xi = train[i][0];
    float yi = train[i][1];
    float zi = train[i][2];
    float ai = sigmoidf(xi*w1 + yi*w2 + b);
    float di = 2*(ai - zi)*ai*(1 - ai);
    *dw1 += di*xi;
    *dw2 += di*yi;
    *db  += di;
  }
  *dw1 /= n;
  *dw2 /= n;
  *db  /= n;
}
  

int main()
{
  // y = x*w + b;

  srand(time(0));
  // srand(69);
  float w1 = rand_float(); 
  float w2 = rand_float();
  float b  = rand_float(); 
  // printf("%f\n", w);

  float rate = 1e-1;

  for (size_t i = 0; i < 100*1000; ++i) {
    float c = cost(w1, w2, b);
    printf("cost: %f, w1: %f, w2: %f, b: %f\n", c, w1, w2, b);

    float dw1, dw2, db;

#if 1
    float eps = 1e-1;
    dcost(eps, w1, w2, b, &dw1, &dw2, &db);
#else
    gcost(w1, w2, b, &dw1, &dw2, &db);
#endif
    w1 -= rate*dw1;
    w2 -= rate*dw2;
    b  -= rate*db;

  }
  printf("--------------------------------------\n");

  printf("w1: %f, w2: %f,  b: %f, cost: %f\n", w1, w2, b, cost(w1, w2, b));

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      printf("%zu %zu -> %f\n", i, j, sigmoidf((float)(w1 * (float)i + w2 * (float)j + b)));
    }
  }

  return 0;
}
