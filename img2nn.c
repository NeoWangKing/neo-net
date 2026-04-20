#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>

#include <raylib.h>
#include <raymath.h>

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

size_t arch[] = {3, 7, 7, 1};

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
                    float thick = 0.005*rh*fabsf((value*2 - 1));
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
        fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
        fprintf(stderr, "ERROR:no image1 file is provided\n");
        return 1;
    }
    const char *img1_file_path = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
        fprintf(stderr, "ERROR:no image2 file is provided\n");
        return 1;
    }
    const char *img2_file_path = args_shift(&argc, &argv);

    int img1_width, img1_height, img1_comp;
    uint8_t *img1_pixels = (uint8_t *)stbi_load(img1_file_path, &img1_width, &img1_height, &img1_comp, 0);
    if (img1_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image %s\n", img1_file_path);
        return 1;
    }
    if (img1_comp != 1) {
        fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bits grayscale images are supported\n", img1_file_path, img1_comp*8);
        return 1;
    }

    int img2_width, img2_height, img2_comp;
    uint8_t *img2_pixels = (uint8_t *)stbi_load(img2_file_path, &img2_width, &img2_height, &img2_comp, 0);
    if (img2_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image %s\n", img2_file_path);
        return 1;
    }
    if (img2_comp != 1) {
        fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bits grayscale images are supported\n", img2_file_path, img2_comp*8);
        return 1;
    }
    
    printf("%s size %dx%d %d bits\n", img1_file_path, img1_width, img1_height, img1_comp*8);
    printf("%s size %dx%d %d bits\n", img2_file_path, img2_width, img2_height, img2_comp*8);

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));

    Mat t = mat_alloc(img1_width*img1_height + img2_width*img2_height, NN_INPUT(nn).cols + NN_OUTPUT(nn).cols);

    for (int y = 0; y < img1_height; ++y) {
        for (int x = 0; x < img1_width; ++x) {
            size_t i = y*img1_width + x;
            MAT_AT(t, i, 0) = (float)x/(img1_width - 1);
            MAT_AT(t, i, 1) = (float)y/(img1_height - 1);
            MAT_AT(t, i, 2) = 0.0f;
            MAT_AT(t, i, 3) = img1_pixels[y*img1_width + x]/255.f;
        }
    }
    for (int y = 0; y < img2_height; ++y) {
        for (int x = 0; x < img2_width; ++x) {
            size_t i = img1_width*img1_height + y*img2_width + x;
            MAT_AT(t, i, 0) = (float)x/(img2_width - 1);
            MAT_AT(t, i, 1) = (float)y/(img2_height - 1);
            MAT_AT(t, i, 2) = 1.0f;
            MAT_AT(t, i, 3) = img2_pixels[y*img2_width + x]/255.f;
        }
    }
    mat_shuffle_rows(t);

    MAT_PRINT(t);

    nn_rand(nn, -1, 1);

    // initialize the components
    size_t WINDOW_FACTOR = 100;
    size_t WINDOW_WIDTH = (16*WINDOW_FACTOR);
    size_t WINDOW_HEIGHT = (9*WINDOW_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "img2mat");
    SetTargetFPS(120);

    Cost_Plot plot = {0};

    Image train_image1 = GenImageColor(img1_width, img1_height, BLACK);
    for (size_t y = 0; y < (size_t)img1_height; ++y) {
        for (size_t x = 0; x < (size_t)img1_width; ++x) {
            uint8_t pixel1 = img1_pixels[y*img1_width + x];
            ImageDrawPixel(&train_image1, x, y, CLITERAL(Color) { pixel1, pixel1, pixel1, 255 });
        }
    }
    Texture2D train_texture1 = LoadTextureFromImage(train_image1);

    Image train_image2 = GenImageColor(img2_width, img2_height, BLACK);
    for (size_t y = 0; y < (size_t)img2_height; ++y) {
        for (size_t x = 0; x < (size_t)img2_width; ++x) {
            uint8_t pixel2 = img2_pixels[y*img2_width + x];
            ImageDrawPixel(&train_image2, x, y, CLITERAL(Color) { pixel2, pixel2, pixel2, 255 });
        }
    }
    Texture2D train_texture2 = LoadTextureFromImage(train_image2);

    int preview_image1_width = 28;
    int preview_image1_height = 28;
    Image preview_image1 = GenImageColor(preview_image1_width, preview_image1_height, BLACK);
    Texture2D preview_texture1 = LoadTextureFromImage(preview_image1);

    int preview_image2_width = 28;
    int preview_image2_height = 28;
    Image preview_image2 = GenImageColor(preview_image2_width, preview_image2_height, BLACK);
    Texture2D preview_texture2 = LoadTextureFromImage(preview_image2);

    int preview_image3_width = 28;
    int preview_image3_height = 28;
    Image preview_image3 = GenImageColor(preview_image3_width, preview_image3_height, BLACK);
    Texture2D preview_texture3 = LoadTextureFromImage(preview_image3);

    size_t epoch = 0;
    size_t epoch_max = 100*1000;
    // size_t epochs_per_frame = 103;
    size_t batches_per_frame = 280;
    size_t batch_size = 28;
    size_t batch_count = (t.rows + batch_size - 1)/batch_size;
    size_t batch_begin = 0;
    float average_cost = 0.f;
    float rate = 0.5f;
    bool paused = true;

    float scroll = 0.5f;
    bool scroll_dragging = false;

    while (!WindowShouldClose()) {
        // some window behaviour
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }

        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        // doing the calculate
        for (size_t i = 0; i < batches_per_frame && !paused && epoch < epoch_max; ++i) {
            size_t size = batch_size;
            if (batch_begin + batch_size >= t.rows) {
                size = t.rows - batch_begin;
            }

            Mat batch_ti = {
                .rows = size,
                .cols = NN_INPUT(nn).cols,
                .stride = t.stride,
                .es = &MAT_AT(t, batch_begin, 0),
            };

            Mat batch_to = {
                .rows = size,
                .cols = NN_OUTPUT(nn).cols,
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
                mat_shuffle_rows(t);
            }
        }
        
        // render the window
        Color background_color = { 0x18, 0x18, 0x18, 0xFF };
        BeginDrawing();
        ClearBackground(background_color);
        {
            for (size_t y = 0; y < (size_t)preview_image1_height; ++y) {
                for (size_t x = 0; x < (size_t)preview_image1_width; ++x) {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(preview_image1_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(preview_image1_height - 1);
                    MAT_AT(NN_INPUT(nn), 0, 2) = 0.0f;
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(&preview_image1, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
                }
            }

            for (size_t y = 0; y < (size_t)preview_image2_height; ++y) {
                for (size_t x = 0; x < (size_t)preview_image2_width; ++x) {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(preview_image2_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(preview_image2_height - 1);
                    MAT_AT(NN_INPUT(nn), 0, 2) = 1.0f;
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(&preview_image2, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
                }
            }

            for (size_t y = 0; y < (size_t)preview_image3_height; ++y) {
                for (size_t x = 0; x < (size_t)preview_image3_width; ++x) {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(preview_image3_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(preview_image3_height - 1);
                    MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(&preview_image3, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
                }
            }
            
            int w, h;
            w = GetRenderWidth();
            h = GetRenderHeight();

            int rw, rh, rx, ry;

            rw = w/3;
            rh = h*2/3;
            rx = 0;
            ry = h/2 - rh/2;
            plot_cost(plot, rx, ry, rw, rh);

            rx += rw;
            nn_render_raylib(nn, rx, ry, rw, rh);

            rx += rw;
            rw /= 2;
            rh = rw;
            if (3*rh > h*2/3) { rh = h*2/9; rw = rh; };
            ry = h/2 - rh*3/2;
            float train_image1_scale = (float)rh/img1_height;
            DrawTextureEx(train_texture1, CLITERAL(Vector2) { rx, ry }, 0, train_image1_scale, WHITE);

            rx += rw;
            float train_image2_scale = (float)rh/img2_height;
            DrawTextureEx(train_texture2, CLITERAL(Vector2) { rx, ry }, 0, train_image2_scale, WHITE);

            float preview_image1_scale = (float)rh/(float)(preview_image1_height);
            rx -= rw;
            ry += rh;
            UpdateTexture(preview_texture1, preview_image1.data);
            DrawTextureEx(preview_texture1, CLITERAL(Vector2) { rx, ry }, 0, preview_image1_scale, WHITE);

            float preview_image2_scale = (float)rh/(float)(preview_image2_height);
            rx += rw;
            UpdateTexture(preview_texture2, preview_image2.data);
            DrawTextureEx(preview_texture2, CLITERAL(Vector2) { rx, ry }, 0, preview_image2_scale, WHITE);

            {
                float preview_image3_scale = (float)rh/(float)(preview_image3_height);
                rx -= rw;
                ry += rh;
                UpdateTexture(preview_texture3, preview_image3.data);
                DrawTextureEx(preview_texture3, CLITERAL(Vector2) { rx + scroll*rw, ry }, 0, preview_image3_scale, WHITE);

                int padding = 10;
                ry += rh + padding;
                rw *= 2;
                rh = 5;
                Vector2 scrollbar_position = { rx, ry };
                Vector2 scrollbar_size = { rw, rh };
                DrawRectangleV(scrollbar_position, scrollbar_size, RAYWHITE);

                ry += rh/2;
                int knob_radius = 10;
                Vector2 knob_position = { rx + scroll * rw, ry };
                DrawCircleV(knob_position, knob_radius, CLITERAL(Color) { 0xFF, 0x55, 0x55, 0xFF });

                if (scroll_dragging) {
                    float x = GetMousePosition().x;
                    if (x < scrollbar_position.x) x = scrollbar_position.x;
                    if (x > scrollbar_position.x + scrollbar_size.x) x = scrollbar_position.x + scrollbar_size.x;
                    scroll = (x - scrollbar_position.x)/scrollbar_size.x;
                }

                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    if (Vector2Distance(GetMousePosition(), knob_position) <= knob_radius) {
                        scroll_dragging = true;
                    }
                }
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                    scroll_dragging = false;
                }
            }

            int font_size = 50*((float)h/(float)WINDOW_HEIGHT);
            char buffers[256];
            snprintf(buffers, sizeof(buffers), "Epoch: %zu/%zu\nCost: %f", epoch, epoch_max, plot.count > 0 ? plot.items[plot.count - 1] : 0); 
            DrawText(buffers, 0, 0, font_size, WHITE);

            // char *status = !paused ? "RUNNING (SPACE to pause)" : "PAUSED (SPACE to start)";
            // DrawText("RESET: R", 0, h - font_size, font_size / 2, LIGHTGRAY);
            // DrawText(status, 0, h - font_size/2, font_size / 2, LIGHTGRAY);
        }
        EndDrawing();
    }

    // for (size_t y = 0; y < (size_t)img1_height; ++y) {
    //     for (size_t x = 0; x < (size_t)img1_width; ++x) {
    //         uint8_t pixel = img1_pixels[y*img1_width + x];
    //         if (pixel) printf("%3u ", pixel); else printf("    ");
    //     }
    //     printf("\n");
    // }
    //
    // for (size_t y = 0; y < (size_t)img1_height; ++y) {
    //     for (size_t x = 0; x < (size_t)img1_width; ++x) {
    //         MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(img1_width - 1);
    //         MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(img1_height - 1);
    //         MAT_AT(NN_INPUT(nn), 0, 2) = 0.0f;
    //         nn_forward(nn);
    //         uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
    //         if (pixel) printf("%3u ", pixel); else printf("    ");
    //     }
    //     printf("\n");
    // }

    // output a upscaled image
    size_t out_width = 512;
    size_t out_height = 512;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels)*out_width*out_height);
    assert(out_pixels != NULL);

    for (size_t y = 0; y < (size_t)out_height; ++y) {
        for (size_t x = 0; x < (size_t)out_width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(out_height - 1);
            MAT_AT(NN_INPUT(nn), 0, 2) = 0.5f;
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

    printf("Generated %s from %s\n", out_file_path, img1_file_path);

    return 0;
}
