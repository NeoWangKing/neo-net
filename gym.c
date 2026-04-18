// Gym is a GUI app that trains your NN on the data you give it.
// The idea is that it will spit out a binary file that can be
// then loaded up with nn.h and used in your application.

#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include "raylib.h"
#define SV_IMPLEMENTATION
#include "sv.h"
#define NN_IMPLEMENTATION
#include "nn.h"

#define IMG_FACTOR 100
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

typedef struct {
    size_t *items;
    size_t count;
    size_t capacity;
} Arch;

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

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

void cost_plot_minmax(Cost_Plot plot, float *min, float *max)
{
    *min = FLT_MIN;
    *max = FLT_MIN;
    for (size_t i = 0; i < plot.count; ++i) {
        if (*max < plot.items[i]) *max = plot.items[i];
        if (*min > plot.items[i]) *min = plot.items[i];
    }
}

void plot_cost(Cost_Plot plot, int rw, int rh, int rx, int ry)
{
    (void) plot;
    float min, max;
    cost_plot_minmax(plot, &min, &max);
    if (min > 0) min = 0;
    size_t n = plot.count;
    if (n < 500) n = 500;
    // DrawLineEx((Vector2){0, ry+rh}, (Vector2){rw, ry+rh}, rh*0.005f, RED);
    for (size_t i = 0; i+1 < plot.count; ++i) {
        float x1 = rx + (float)rw/n*i;
        float y1 = ry + (1.0f - (plot.items[i] - min)/(max - min))*rh;
        float x2 = rx + (float)rw/n*(i+1);
        float y2 = ry + (1.0f - (plot.items[i+1] - min)/(max - min))*rh;
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh*0.005f, RED);
    }
}

void nn_render_raylib(NN nn, int rw, int rh, int rx, int ry)
{
    Color neuron_color_inactivated = { 0x5F, 0x5F, 0x5F, 0xFF };
    Color low_color = { 0xFF, 0x55, 0x55, 0xFF };
    Color high_color = { 0x55, 0xFF, 0xFF, 0xFF };

    float neuron_radius = rh*0.04;
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
                    float value = sigmoidf(MAT_AT(nn.ws[l], j, i));
                    high_color.a = floorf(255.f*value);
                    float thick = 0.004*rh*fabs((value*2 - 1));
                    Vector2 start = { cx1, cy1 };
                    Vector2 end   = { cx2, cy2 };
                    DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
                    // DrawLine(cx1, cy1, cx2, cy2, ColorAlphaBlend(low_color, high_color, WHITE));
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

int main(int argc, char **argv)
{
    const char *program = args_shift(&argc, &argv);

    if(argc <= 0) {
        fprintf(stderr, "Usage: %s\n <model.arch> <model.mat>", program);
        fprintf(stderr, "ERROR: no architecture file was provided\n");
        return 1;
    }
    const char *arch_file_path = args_shift(&argc, &argv);

    if(argc <= 0) {
        fprintf(stderr, "Usage: %s\n <model.arch> <model.mat>", program);
        fprintf(stderr, "ERROR: no data file was provided\n");
        return 1;
    }
    const char *data_file_path = args_shift(&argc, &argv);

    int buffer_len = 0;
    unsigned char *buffer = LoadFileData(arch_file_path, &buffer_len);
    if (buffer == NULL) {
        return 1;
    }

    String_View content = sv_from_parts((const char*)buffer, buffer_len);
    Arch arch = {0};
    content = sv_trim_left(content);
    while (content.count > 0 && isdigit(content.data[0])) {
        size_t x = sv_chop_u64(&content);
        da_append(&arch, x);
        // printf("%d\n", x);
        content = sv_trim_left(content);
    }

    FILE *in = fopen(data_file_path, "rb");
    if (in == NULL) {
        fprintf(stderr, "ERROR: could not read file %s\n", data_file_path);
        return 1;
    }
    Mat t = mat_load(in);
    fclose(in);

    // MAT_PRINT(t);

    // can we have NN with just input?
    NN_ASSERT(arch.count > 1);
    size_t ins_sz = arch.items[0];
    size_t outs_sz = arch.items[arch.count-1];
    NN_ASSERT(t.cols == ins_sz + outs_sz);

    Mat ti = {
        .rows = t.rows,
        .cols = ins_sz,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, 0),
    };

    Mat to = {
        .rows = t.rows,
        .cols = outs_sz,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, ins_sz),
    };

    // MAT_PRINT(ti);
    // MAT_PRINT(to);

    NN nn = nn_alloc(arch.items, arch.count);
    NN g = nn_alloc(arch.items, arch.count);
    nn_rand(nn, 0, 1);
    NN_PRINT(nn);

    float rate = 1;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
    SetTargetFPS(120);
    Cost_Plot plot = {0};

    size_t epoch = 0;
    size_t epoch_max = 10*1000;
    char buffers[256];
    int rw, rh, rx, ry;
    int w, h;
    int font_size = 50;

    while (!WindowShouldClose()) {
        for (size_t i = 0; i < 10 && epoch < epoch_max; ++i) {
            if (epoch < epoch_max) {
                nn_backprop(nn, g, ti, to);
                nn_learn(nn, g, rate);
                epoch += 1;
                da_append(&plot, nn_cost(nn, ti, to));
            }
        }
        
        Color background_color = { 0x18, 0x18, 0x18, 0xFF };
        BeginDrawing();
        ClearBackground(background_color);
        {
            w = GetRenderWidth();
            h = GetRenderHeight();
            font_size = 50*((float)h/(float)IMG_HEIGHT);

            snprintf(buffers, sizeof(buffers), "Epoch: %zu/%zu\nCost: %f", epoch, epoch_max, nn_cost(nn, ti, to)); 
            DrawText(buffers, 0, 0, font_size, WHITE);

            rw = w/2;
            rh = h*2/3;
            rx = 0;
            ry = h/2 - rh/2;
            plot_cost(plot, rw, rh, rx, ry);

            rw = w/2;
            rh = h*2/3;
            rx = w - rw;
            ry = h/2 - rh/2;
            nn_render_raylib(nn, rw, rh, rx, ry);
        }
        EndDrawing();
    }

    // NN_PRINT(nn);

    return 0;
}
