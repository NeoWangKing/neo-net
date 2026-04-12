#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

float rand_float(void);
float sigmoidf(float x);

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat   mat_alloc(size_t rows, size_t cols);
void  mat_rand(Mat dst, float low, float high);
Mat   mat_sub(Mat src, size_t rows, size_t cols, size_t r, size_t c);
Mat   mat_row(Mat src, size_t rows);
Mat   mat_col(Mat src, size_t cols);
void  mat_copy(Mat dst, Mat src);
Mat   mat_def(size_t rows, size_t cols, float *es);
void  mat_unit(Mat dst);
void  mat_fill(Mat dst, float x);
void  mat_times(Mat dst, float x);
void  mat_sum(Mat dst, Mat a);
void  mat_dot(Mat dst, Mat a, Mat b);
void  mat_sig(Mat dst);
void  mat_print(Mat dst, const char *name, size_t padding);

#define MAT_PRINT(m) mat_print(m, #m, 0)

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

typedef struct {
  size_t count;
  Mat *ws;
  Mat *bs;
  Mat *as;
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void  nn_rand(NN nn, float low, float high);
void  nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void  nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void  nn_learn(NN nn, NN g, float rate);
void  nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn)

#endif // NN_H_

// ==================================================================================

#ifdef NN_IMPLEMENTATION

float rand_float(void)
{
  return ((float) rand() / (float) RAND_MAX);
}

float sigmoidf(float x)
{
  return 1.f / (1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols)
{
  Mat m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.es = NN_MALLOC(sizeof(*m.es));
  NN_ASSERT(m.es != NULL);
  return m;
}

Mat mat_sub(Mat src, size_t rows, size_t cols, size_t r, size_t c)
{
  NN_ASSERT(r + rows <= src.rows);
  NN_ASSERT(c + cols <= src.cols);

  return (Mat){
    .rows = rows,
    .cols = cols,
    .stride = src.stride,
    .es = &MAT_AT(src, r, c),
  };
}

Mat mat_row(Mat src, size_t rows)
{
  return mat_sub(src, 1, src.cols, rows, 0);
}

Mat mat_col(Mat src, size_t cols)
{
  return mat_sub(src, src.rows, 1, 0, cols);
}

void mat_copy(Mat dst, Mat src)
{
  NN_ASSERT(dst.rows == src.rows);
  NN_ASSERT(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}

Mat mat_def(size_t rows, size_t cols, float *es)
{
  NN_ASSERT(es != NULL);
  Mat m = mat_alloc(rows, cols);
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = es[i*rows + j];
    }
  }
  return m;
}

void mat_rand(Mat dst, float low, float high)
{
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = rand_float()*(high - low) + low;
    }
  }
}

void mat_unit(Mat dst)
{
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      if (i == j) {
        MAT_AT(dst, i, j) = 1.f;
      }else {
        MAT_AT(dst, i, j) = 0.f;
      }
    }
  }
}

void mat_fill(Mat dst, float x)
{
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = x;
    }
  }
}

void mat_times(Mat dst, float x) {
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) *= x;
    }
  }
}

void mat_sum(Mat dst, Mat a)
{
  NN_ASSERT(dst.rows == a.rows);
  NN_ASSERT(dst.cols == a.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) += MAT_AT(a, i, j);
    }
  }
}

void mat_dot(Mat dst, Mat a, Mat b)
{
  NN_ASSERT(a.cols == b.rows);
  NN_ASSERT(dst.rows == a.rows);
  NN_ASSERT(dst.cols == b.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = 0;
    }
  }
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      for (size_t k = 0; k < a.cols; ++k) {
        // (i * k)  (k * j) => (i * j)
        //      ^    ^
        MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
        //                               ^              ^
      }
    }
  }
}

void mat_sig(Mat dst)
{
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = sigmoidf(MAT_AT(dst, i, j));
    }
  }
}

void mat_print(Mat dst, const char *name, size_t padding)
{
  printf("%*s%s = [\n", (int) padding, "", name);
  for (size_t i = 0; i < dst.rows; ++i) {
    printf("%*s    ", (int)padding, "");
    for (size_t j = 0; j < dst.cols; ++j) {
      printf("%f ", MAT_AT(dst, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int) padding, "");
}

// ==================================================================================

NN nn_alloc(size_t *arch, size_t arch_count)
{
  NN_ASSERT(arch_count > 0);
  NN nn;
  nn.count = arch_count - 1;

  nn.ws = malloc(sizeof(*nn.ws)*nn.count);
  NN_ASSERT(nn.ws != NULL);
  nn.bs = malloc(sizeof(*nn.bs)*nn.count);
  NN_ASSERT(nn.bs != NULL);
  nn.as = malloc(sizeof(*nn.as)*(nn.count + 1));
  NN_ASSERT(nn.as != NULL);

  nn.as[0] = mat_alloc(1, arch[0]);
  for (size_t i = 1; i < arch_count; ++i) {
    nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
    nn.bs[i-1] = mat_alloc(1, arch[i]);
    nn.as[i]   = mat_alloc(1, arch[i]);
  }

  return nn;
}

void nn_print(NN nn, const char *name)
{
  char buf[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < nn.count; ++i) {
    snprintf(buf, sizeof(buf), "ws[%zu]", i);
    mat_print(nn.ws[i], buf, 4);
    snprintf(buf, sizeof(buf), "bs[%zu]", i);
    mat_print(nn.bs[i], buf, 4);
  }
  printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{

  for (size_t i = 0; i < nn.count; ++i) {
    mat_rand(nn.ws[i], low, high);
    mat_rand(nn.bs[i], low, high);
  }
}

void nn_forward(NN nn)
{
  for (size_t i = 0; i < nn.count; ++i) {
    mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
    mat_sum(nn.as[i+1], nn.bs[i]);
    mat_sig(nn.as[i+1]);
  }
}

float nn_cost(NN nn, Mat ti, Mat to)
{
  NN_ASSERT(ti.rows == to.rows);
  NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
  size_t n = ti.rows;

  float c = 0.f;
  for (size_t i = 0; i < n; ++i) {
    Mat EXPECT_IN  = mat_row(ti, i);
    Mat EXPECT_OUT = mat_row(to, i);

    mat_copy(NN_INPUT(nn), EXPECT_IN);
    nn_forward(nn);

    size_t m = to.cols;
    for (size_t j = 0; j < m; ++j) {
      float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(EXPECT_OUT, 0, j);
      c += d*d;
    }
  }

  return c/n;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to)
{
  float saved;
  float c = nn_cost(nn, ti, to);
  for (size_t i = 0; i < nn.count; ++i) {
    for (size_t j = 0; j < nn.ws[i].rows; ++j) {
      for (size_t k = 0; k < nn.ws[i].cols; ++k) {
        saved = MAT_AT(nn.ws[i], j, k);
        MAT_AT(nn.ws[i], j, k) += eps;
        MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c)/eps;
        MAT_AT(nn.ws[i], j, k) = saved;
      }
    }
    for (size_t j = 0; j < nn.bs[i].rows; ++j) {
      for (size_t k = 0; k < nn.bs[i].cols; ++k) {
        saved = MAT_AT(nn.bs[i], j, k);
        MAT_AT(nn.bs[i], j, k) += eps;
        MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c)/eps;
        MAT_AT(nn.bs[i], j, k) = saved;
      }
    }
  }
}

void nn_learn(NN nn, NN g, float rate)
{
  for (size_t i = 0; i < nn.count; ++i) {
    for (size_t j = 0; j < nn.ws[i].rows; ++j) {
      for (size_t k = 0; k < nn.ws[i].cols; ++k) {
        MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
      }
    }
    for (size_t j = 0; j < nn.bs[i].rows; ++j) {
      for (size_t k = 0; k < nn.bs[i].cols; ++k) {
        MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
      }
    }
  }
}

#endif // NN_IMPLEMANTATION
