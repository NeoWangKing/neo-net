#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <float.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <raylib.h>
#include <raymath.h>

// #define STB_IMAGE_IMPLEMENTATION
#include "thirdparty/stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb_image_write.h"

#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

#define FPS 60
#define STR2(x) #x
#define STR(x) STR2(x)
#define READ_END 0
#define WRITE_END 1

size_t arch[] = {3, 11, 11, 9, 1};
size_t epoch_max = 100*1000;
size_t batches_per_frame = 280;
size_t batch_size = 28;
float rate = 1.0f;
float scroll = 0.f;

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

#define OUT_WIDTH 256
#define OUT_HEIGHT 256
uint32_t out_pixels[OUT_WIDTH*OUT_HEIGHT];

float transition(float i, float start, float end)
{
    float j = i;
    if (i < 0.5f) j = 2*i*i;
    if (i >= 0.5f) j = 1 - 2*(1-i)*(1-i);
    float a = j*(end - start) + start;
    return a;
}

void render_single_out_image(NN nn, float scroll)
{
    for (size_t y = 0; y < (size_t)OUT_HEIGHT; ++y) {
        for (size_t x = 0; x < (size_t)OUT_WIDTH; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(OUT_WIDTH - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(OUT_HEIGHT - 1);
            MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
            nn_forward(nn);
            float activation = MAT_AT(NN_OUTPUT(nn), 0, 0);
            if (activation < 0) activation = 0;
            if (activation > 1) activation = 1;
            uint32_t bright = activation*255.f;
            uint32_t pixel = 0xFF000000|bright|(bright<<8)|(bright<<16);
            out_pixels[y*OUT_WIDTH + x] = pixel;
        }
    }
}

int render_upscaled_screenshot(NN nn, const char *out_file_path, float scroll)
{
    render_single_out_image(nn, scroll);

    if (!stbi_write_png(out_file_path, OUT_WIDTH, OUT_HEIGHT, 4, out_pixels, OUT_WIDTH*sizeof(*out_pixels))) {
        fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
        return 1;
    }

    printf("Generated %s!\n", out_file_path);
    return 0;
}

int render_upscaled_video(NN nn, const char *out_file_path, float duration)
{
    int pipefd[2];

    if (pipe(pipefd) < 0) {
        fprintf(stderr, "ERROR: could not create a pipe: %s\n", strerror(errno));
        return 1;
    }
    
    pid_t child = fork();
    if (child < 0) {
        fprintf(stderr, "ERROR: could not fork a child: %s\n", strerror(errno));
        return 1;
    }
    if (child == 0) {
        if (dup2(pipefd[READ_END], STDIN_FILENO) < 0) {
            fprintf(stderr, "ERROR: could not reopen read end of pipe as stdin: %s\n", strerror(errno));
            return 1;
        }
        close(pipefd[WRITE_END]);
        
        int ret = execlp("ffmpeg",
                "ffmpeg",
                "-loglevel", "verbose",
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", STR(OUT_WIDTH) "x" STR(OUT_HEIGHT),
                "-r", STR(FPS),
                "-an",
                "-i", "-",
                "-c:v", "libx264",

                out_file_path,
                // ...
                NULL
              );
        if (ret < 0) {
            fprintf(stderr, "ERROR: could not run ffmpeg as a child process: %s\n", strerror(errno));
            return 1;
        }
        assert(0 && "unreachable");
    }

    close(pipefd[READ_END]);

    typedef struct {
        float start;
        float end;
        float duration;
    } Segment;

    Segment segments[] = {
        {0, 0, 1},
        {0, 1, 1},
        {1, 1, 1},
        {1, 0, 1},
        {0, 0, 1},
    };
    size_t segments_count = ARRAY_LEN(segments);
    float segments_total_duration = 0;

    for (size_t i = 0; i < segments_count; ++i) {
        segments_total_duration += segments[i].duration;
    }

    for (size_t i = 0; i < segments_count; ++i) {
        size_t frame_count = FPS*(segments[i].duration/segments_total_duration)*duration;
        for (size_t j = 0; j < frame_count; ++j) {
            render_single_out_image(nn, transition((float)j/frame_count, segments[i].start, segments[i].end));
            ssize_t written = write(pipefd[WRITE_END], out_pixels, sizeof(*out_pixels)*OUT_WIDTH*OUT_HEIGHT);
            if (written != (ssize_t)(sizeof(*out_pixels)*OUT_WIDTH*OUT_HEIGHT)) {
                fprintf(stderr, "ERROR:could not render frame (write returned %zd)\n", written);
            }
        }
    }

    close(pipefd[WRITE_END]);
    wait(NULL);
    printf("Generated %s!\n", out_file_path);
    return 0;
}

int main(int argc, char **argv)
{
    const char *program = args_shift(&argc, &argv);

    // deal width the args
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

    nn_rand(nn, -1, 1);

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

    // MAT_PRINT(t);

    // initialize the components
    size_t WINDOW_FACTOR = 100;
    size_t WINDOW_WIDTH = (16*WINDOW_FACTOR);
    size_t WINDOW_HEIGHT = (9*WINDOW_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "img2mat");
    SetTargetFPS(120);

    Font font = LoadFontEx("./font/JetBrainsMonoNerdFont-Medium.ttf", 50, 0, 250);
    SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);

    Gym_Plot plot = {0};

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

    Gym_Batch gb = {0};

    bool paused = true;
    bool scroll_dragging = false;
    bool rate_dragging = false;
    size_t epoch = 0;

    while (!WindowShouldClose()) {
        // some keyboard behaviour
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }

        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        if (IsKeyPressed(KEY_S)) {
            render_upscaled_screenshot(nn, "upscaled.png", scroll);
        }

        if (IsKeyPressed(KEY_X)) {
            render_upscaled_video(nn, "upscaled.mp4", 5);
        }

        // doing the calculate
        for (size_t i = 0; i < batches_per_frame && !paused && epoch < epoch_max; ++i) {
            gym_process_batch(&gb, batch_size, nn, g, t, rate);
            if (gb.finished) {
                epoch += 1;
                da_append(&plot, gb.cost);
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
            gym_plot_cost(plot, rx, ry, rw, rh);

            rx += rw;
            gym_render_nn(nn, rx, ry, rw, rh);

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
                float pad = rh*0.05;
                ry = ry + rh + pad;
                rw *= 2;
                rh = rh*0.02;
                gym_slider(&scroll, &scroll_dragging, rx, ry, rw, rh);
            }

            int font_size = 50*((float)h/(float)WINDOW_HEIGHT);
            char buffers[256];
            snprintf(buffers, sizeof(buffers), "Epoch: %zu/%zu, Rate: %f, Cost: %f", epoch, epoch_max, rate, plot.count > 0 ? plot.items[plot.count - 1] : 0); 
            DrawTextEx(font, buffers, CLITERAL(Vector2){ 0, 0 }, font_size, 0, WHITE);
            gym_slider(&rate, &rate_dragging, 0, h*0.08, w, h*0.02);

            char *status = !paused ? "RUNNING (SPACE to pause)" : "PAUSED (SPACE to start)";
            DrawTextEx(font, "RESET: [R], SAVE: [S]", CLITERAL(Vector2){ 0, h - font_size*2 }, font_size, 0, LIGHTGRAY);
            DrawTextEx(font, status, CLITERAL(Vector2){ 0, h - font_size }, font_size, 0, LIGHTGRAY);
        }
        EndDrawing();
    }

    return 0;
}
