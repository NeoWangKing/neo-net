#include <stdio.h>
#include <assert.h>
#include <float.h>

#include <raylib.h>

// #define STB_IMAGE_IMPLEMENTATION
#include "thirdparty/stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

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
    *min = FLT_MAX;
    *max = -FLT_MAX;
    for (size_t i = 0; i < plot.count; ++i) {
        if (*max < plot.items[i]) *max = plot.items[i];
        if (*min > plot.items[i]) *min = plot.items[i];
    }
}

void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh)
{
    (void) plot;
    float min, max;
    cost_plot_minmax(plot, &min, &max);
    if (min > 0) min = 0;
    size_t n = plot.count;
    if (n < 500) n = 500;
    DrawLineEx((Vector2){0, ry+rh}, (Vector2){rw, ry+rh}, rh*0.005f, WHITE);
    DrawText("0", 0, ry+rh, 25, WHITE);
    for (size_t i = 0; i+1 < plot.count; ++i) {
        float x1 = rx + (float)rw/n*i;
        float y1 = ry + (1.0f - (plot.items[i] - min)/(max - min))*rh;
        float x2 = rx + (float)rw/n*(i+1);
        float y2 = ry + (1.0f - (plot.items[i+1] - min)/(max - min))*rh;
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh*0.005f, RED);
    }
}

void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh)
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
                    float value = sigmoidf(MAT_AT(nn.ws[l], i, j));
                    high_color.a = floorf(255.f*value);
                    float thick = 0.005*rh*fabs((value*2 - 1));
                    // float thick = 0.005*rh;
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

int main(int argc, char **argv)
{
    const char *program = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <input>\n", program);
        fprintf(stderr, "ERROR:no input file is provided\n");
        return 1;
    }

    const char *img_file_path = args_shift(&argc, &argv);

    int img_width, img_height, img_comp;
    uint8_t *img_pixels = (uint8_t *)stbi_load(img_file_path, &img_width, &img_height, &img_comp, 0);
    if (img_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image %s\n", img_file_path);
        return 1;
    }
    if (img_comp != 1) {
        fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bits grayscale images are supported\n", img_file_path, img_comp*8);
        return 1;
    }
    
    printf("%s size %dx%d %d bits\n", img_file_path, img_width, img_height, img_comp*8);

    Mat t = mat_alloc(img_width*img_height, 3);

    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            size_t i = y*img_width + x;
            MAT_AT(t, i, 0) = (float)x/(img_width - 1);
            MAT_AT(t, i, 1) = (float)y/(img_height - 1);
            MAT_AT(t, i, 2) = img_pixels[i]/255.f;
        }
    }
    mat_shuffle_rows(t);

    MAT_PRINT(t);

    size_t arch[] = {2, 7, 7, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    size_t WINDOW_FACTOR = 100;
    size_t WINDOW_WIDTH = (16*WINDOW_FACTOR);
    size_t WINDOW_HEIGHT = (9*WINDOW_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "img2mat");
    SetTargetFPS(120);

    Cost_Plot plot = {0};

    Image train_image = GenImageColor(img_width, img_height, BLACK);
    Texture2D train_texture = LoadTextureFromImage(train_image);
    for (size_t y = 0; y < (size_t)img_height; ++y) {
        for (size_t x = 0; x < (size_t)img_width; ++x) {
            uint8_t pixel = img_pixels[y*img_width + x];
            ImageDrawPixel(&train_image, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
        }
    }
    UpdateTexture(train_texture, train_image.data);
    int preview_width = 28;
    int preview_height = 28;
    Image preview_image = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture = LoadTextureFromImage(preview_image);

    size_t epoch = 0;
    size_t epoch_max = 100*1000;
    size_t epochs_per_frame = 103;
    size_t batches_per_frame = 280;
    size_t batch_size = 28;
    size_t batch_count = (t.rows + batch_size - 1)/batch_size;
    size_t batch_begin = 0;
    float average_cost = 0.f;
    float rate = 0.5f;


    bool paused = true;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }

        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        for (size_t i = 0; i < batches_per_frame && !paused && epoch < epoch_max; ++i) {
            size_t size = batch_size;
            if (batch_begin + batch_size >= t.rows) {
                size = t.rows - batch_begin;
            }

            Mat batch_ti = {
                .rows = size,
                .cols = 2,
                .stride = t.stride,
                .es = &MAT_AT(t, batch_begin, 0),
            };

            Mat batch_to = {
                .rows = size,
                .cols = 1,
                .stride = t.stride,
                .es = &MAT_AT(t, batch_begin, batch_ti.cols),
            };

            nn_backprop(nn, g, batch_ti, batch_to);
            nn_learn(nn, g, rate);
            average_cost += nn_cost(nn, batch_ti, batch_to);
            batch_begin += batch_size;

            if (batch_begin >= t.rows) {
                epoch += 1;
                da_append(&plot, average_cost/batch_count);
                average_cost = 0.0f;
                batch_begin = 0;
                // mat_shuffle_rows(t);
            }
        }
        
        Color background_color = { 0x18, 0x18, 0x18, 0xFF };
        BeginDrawing();
        ClearBackground(background_color);
        {
            int w, h;
            w = GetRenderWidth();
            h = GetRenderHeight();

            int font_size = 50*((float)h/(float)WINDOW_HEIGHT);

            char buffers[256];
            snprintf(buffers, sizeof(buffers), "Epoch: %zu/%zu\nCost: %f", epoch, epoch_max, plot.count > 0 ? plot.items[plot.count - 1] : 0); 
            DrawText(buffers, 0, 0, font_size, WHITE);

            // char *status = !paused ? "RUNNING (SPACE to pause)" : "PAUSED (SPACE to start)";
            // DrawText("RESET: R", 0, h - font_size, font_size / 2, LIGHTGRAY);
            // DrawText(status, 0, h - font_size/2, font_size / 2, LIGHTGRAY);
            
            int rw, rh, rx, ry;

            rw = w/3;
            rh = h*2/3;
            rx = 0;
            ry = h/2 - rh/2;
            plot_cost(plot, rx, ry, rw, rh);

            rx += rw;
            nn_render_raylib(nn, rx, ry, rw, rh);

            float scale = (float)h/56;
            rx += rw;
            ry = 0;
            rh = rw;
            DrawTextureEx(train_texture, CLITERAL(Vector2) { rx, ry }, 0, scale, WHITE);

            float preview_scale = (float)h/(float)(2*preview_height);
            ry = h/2;
            for (size_t y = 0; y < (size_t)preview_height; ++y) {
                for (size_t x = 0; x < (size_t)preview_width; ++x) {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(preview_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(preview_height - 1);
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(&preview_image, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
                }
            }
            UpdateTexture(preview_texture, preview_image.data);
            DrawTextureEx(preview_texture, CLITERAL(Vector2) { rx, ry }, 0, preview_scale, WHITE);
        }
        EndDrawing();
    }
    // printf("cost = %f\n", nn_cost(nn, ti, to));

    for (size_t y = 0; y < (size_t)img_height; ++y) {
        for (size_t x = 0; x < (size_t)img_width; ++x) {
            uint8_t pixel = img_pixels[y*img_width + x];
            if (pixel) printf("%3u ", pixel); else printf("    ");
        }
        printf("\n");
    }

    for (size_t y = 0; y < (size_t)img_height; ++y) {
        for (size_t x = 0; x < (size_t)img_width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(img_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(img_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            if (pixel) printf("%3u ", pixel); else printf("    ");
        }
        printf("\n");
    }

    size_t out_width = 512;
    size_t out_height = 512;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels)*out_width*out_height);
    assert(out_pixels != NULL);

    for (size_t y = 0; y < (size_t)out_height; ++y) {
        for (size_t x = 0; x < (size_t)out_width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(out_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            out_pixels[y*out_width + x] = pixel;
        }
    }

    const char *out_file_path = "upscaled.png";
    if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width*sizeof(*out_pixels))) {
        fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
        return 1;
    }

    printf("Generated %s from %s\n", out_file_path, img_file_path);

    return 0;
}
