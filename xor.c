// xor.c —— 交互式 XOR 训练程序（基于 raylib）
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <time.h>

#include <raylib.h>

#define NN_IMPLEMENTATION
#include "nn.h"

// ========== 动态数组（沿用原框架） ==========
typedef struct {
    float *items;
    size_t count;
    size_t capacity;
} Cost_Plot;

#define DA_INIT_CAP 256
#define da_append(da, item)                                                         \
    do {                                                                            \
        if ((da)->count >= (da)->capacity) {                                        \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;  \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items));\
            assert((da)->items != NULL && "Buy more RAM lol");                      \
        }                                                                           \
                                                                                    \
        (da)->items[(da)->count++] = (item);                                        \
    } while (0)

// ========== XOR 数据集（固定） ==========
float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

// ========== Cost 曲线相关函数（从 img2mat.c 移植） ==========
void cost_plot_minmax(Cost_Plot plot, float *min, float *max)
{
    *min = FLT_MAX;
    *max = -FLT_MAX;
    for (size_t i = 0; i < plot.count; ++i) {
        if (*max < plot.items[i]) *max = plot.items[i];
        if (*min > plot.items[i]) *min = plot.items[i];
    }
}

void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh)
{
    float min, max;
    cost_plot_minmax(plot, &min, &max);
    if (min > 0) min = 0;
    size_t n = plot.count;
    if (n < 100) n = 100;   // [修改] 适合 XOR 快速收敛
    DrawLineEx((Vector2){rx, ry+rh}, (Vector2){rx+rw, ry+rh}, rh*0.005f, WHITE);
    DrawText("0", rx, ry+rh, 20, WHITE);
    for (size_t i = 0; i+1 < plot.count; ++i) {
        float x1 = rx + (float)rw/n * i;
        float y1 = ry + (1.0f - (plot.items[i] - min)/(max - min)) * rh;
        float x2 = rx + (float)rw/n * (i+1);
        float y2 = ry + (1.0f - (plot.items[i+1] - min)/(max - min)) * rh;
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh*0.005f, RED);
    }
}

// ========== 网络可视化（从 img2mat.c 移植） ==========
void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh)
{
    Color neuron_color_inactivated = { 0x5F, 0x5F, 0x5F, 0xFF };
    Color low_color = { 0xFF, 0x55, 0x55, 0xFF };
    Color high_color = { 0x55, 0xFF, 0xFF, 0xFF };

    float neuron_radius = rh*0.08;   // [修改] XOR 网络小，神经元画大一点
    int layer_border_vpad = rh*0.08;
    int layer_border_hpad = rw*0.08;

    int nn_width = rw - 2*layer_border_hpad;
    int nn_height = rh - 2*layer_border_vpad;
    int nn_x = rx + rw/2 - nn_width/2;
    int nn_y = ry + rh/2 - nn_height/2;
    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count;
    for (size_t l = 0; l < arch_count; ++l) {
        int layer_vpad1 = nn_height/nn.as[l].cols;
        for (size_t i = 0; i < nn.as[l].cols; ++i) {
            int cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            int cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
            if (l+1 < arch_count) {
                int layer_vpad2 = nn_height/nn.as[l+1].cols;
                for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
                    int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
                    int cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    float value = sigmoidf(MAT_AT(nn.ws[l], i, j));
                    high_color.a = floorf(255.f*value);
                    float thick = rh*0.01f * fabsf(value*2 - 1);   // [修改] 加粗线条便于观察
                    Vector2 start = { cx1, cy1 };
                    Vector2 end   = { cx2, cy2 };
                    DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }
            if (l > 0) {
                high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
                DrawCircle(cx1, cy1, (int)neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
            } else {
                DrawCircle(cx1, cy1, (int)neuron_radius, neuron_color_inactivated);
            }
        }
    }
}

// ========== 主函数 ==========
int main(void)
{
    srand(time(0));
    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    // [修改] 使用 raylib 窗口，固定大小
    const size_t WINDOW_FACTOR = 100;
    const size_t WINDOW_WIDTH = 16*WINDOW_FACTOR;
    const size_t WINDOW_HEIGHT = 9*WINDOW_FACTOR;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "XOR Training");
    SetTargetFPS(60);

    // [沿用] 构建 XOR 数据集矩阵
    size_t arch_count = ARRAY_LEN(arch);
    size_t input_num = arch[0];
    size_t output_num = arch[arch_count - 1];
    size_t stride = input_num + output_num;
    size_t n = sizeof(td)/sizeof(td[0])/stride;

    Mat ti = { .rows = n, .cols = input_num, .stride = stride, .es = td };
    Mat to = { .rows = n, .cols = output_num, .stride = stride, .es = td + input_num };


    // [修改] 使用 Xavier 初始化，范围适合 Sigmoid
    // for (size_t i = 0; i < nn.count; ++i) {
    //     float limit = sqrtf(6.0f / (nn.ws[i].rows + nn.ws[i].cols));
    //     mat_rand(nn.ws[i], -limit, limit);
    //     mat_rand(nn.bs[i], -limit, limit);
    // }

    float rate = 0.5f;          // [修改] 学习率调整
    size_t epoch = 0;
    size_t epoch_max = 10000;   // [修改] 最大 epoch 数
    bool paused = true;

    Cost_Plot plot = {0};

    // [新增] 用于显示真值表的纹理（简单文本即可，这里用 DrawText 直接绘制）

    while (!WindowShouldClose()) {
        // 键盘交互
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            plot.count = 0;
            // 重新初始化
            for (size_t i = 0; i < nn.count; ++i) {
                float limit = sqrtf(6.0f / (nn.ws[i].rows + nn.ws[i].cols));
                mat_rand(nn.ws[i], -limit, limit);
                mat_rand(nn.bs[i], -limit, limit);
            }
        }

        // 训练循环（每帧 50 次迭代）
        if (!paused && epoch < epoch_max) {
            for (int iter = 0; iter < 50; ++iter) {
                nn_backprop(nn, g, ti, to);
                nn_learn(nn, g, rate);
                epoch++;
                float cost = nn_cost(nn, ti, to);
                da_append(&plot, cost);
                if (epoch >= epoch_max) break;
            }
        }

        BeginDrawing();
        ClearBackground((Color){0x18, 0x18, 0x18, 0xFF});

        int w = GetRenderWidth();
        int h = GetRenderHeight();
        int font_size = 30 * h / WINDOW_HEIGHT;

        // 状态栏文字
        char buf[256];
        float current_cost = plot.count > 0 ? plot.items[plot.count-1] : nn_cost(nn, ti, to);
        snprintf(buf, sizeof(buf), "Epoch: %zu/%zu, Cost: %.6f", epoch, epoch_max, current_cost);
        DrawText(buf, 10, 10, font_size, WHITE);

        const char *status = paused ? "PAUSED (SPACE to run)" : "RUNNING (SPACE to pause)";
        DrawText(status, 10, h - font_size, font_size/2, LIGHTGRAY);
        DrawText("RESET: R", 10, h - font_size*2, font_size/2, LIGHTGRAY);

        // 布局：左侧 Cost 曲线，中间网络结构，右侧真值表
        int rw = w/3;
        int rh = h*2/3;
        int rx = 0;
        int ry = h/2 - rh/2;

        plot_cost(plot, rx, ry, rw, rh);

        rx += rw;
        nn_render_raylib(nn, rx, ry, rw, rh);

        // [新增] 右侧显示 XOR 真值表及当前输出
        rx += rw;
        int table_x = rx + 20;
        int table_y = ry + 20;
        int line_height = font_size * 1.2f;

        DrawText("XOR Truth Table", table_x, table_y, font_size, WHITE);
        table_y += line_height * 1.5;

        // 对每个输入组合计算网络输出并显示
        for (size_t combo = 0; combo < n; ++combo) {
            float in1 = MAT_AT(ti, combo, 0);
            float in2 = MAT_AT(ti, combo, 1);
            float expected = MAT_AT(to, combo, 0);

            // 前向传播
            MAT_AT(NN_INPUT(nn), 0, 0) = in1;
            MAT_AT(NN_INPUT(nn), 0, 1) = in2;
            nn_forward(nn);
            float predicted = MAT_AT(NN_OUTPUT(nn), 0, 0);

            snprintf(buf, sizeof(buf), "%d XOR %d = %.4f (target: %d)",
                     (int)in1, (int)in2, predicted, (int)expected);
            Color text_color = (fabsf(predicted - expected) < 0.5f) ? GREEN : RED;
            DrawText(buf, table_x, table_y, font_size/1.5, text_color);
            table_y += line_height;
        }

        // 显示网络参数数量（小提示）
        snprintf(buf, sizeof(buf), "Params: %zu", nn.count > 0 ? (nn.ws[0].rows*nn.ws[0].cols + nn.bs[0].cols) : 0);
        DrawText(buf, table_x, table_y + line_height, font_size/2, LIGHTGRAY);

        EndDrawing();
    }

    CloseWindow();

    // 控制台输出最终结果
    printf("===========================================\n");
    for (size_t combo = 0; combo < n; ++combo) {
        float in1 = MAT_AT(ti, combo, 0);
        float in2 = MAT_AT(ti, combo, 1);
        float expected = MAT_AT(to, combo, 0);
        MAT_AT(NN_INPUT(nn), 0, 0) = in1;
        MAT_AT(NN_INPUT(nn), 0, 1) = in2;
        nn_forward(nn);
        float predicted = MAT_AT(NN_OUTPUT(nn), 0, 0);
        printf("%.0f XOR %.0f = %.4f (target: %.0f)\n", in1, in2, predicted, expected);
    }

    return 0;
}
