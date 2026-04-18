// xor.c
#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"
#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMG_WIDTH 800
#define IMG_HEIGHT 600

uint32_t img_pixels[IMG_WIDTH*IMG_HEIGHT];

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

void nn_render(Olivec_Canvas img, NN nn)
{
    uint32_t neuron_color_inactivated = 0xFF5F5F5F;
    uint32_t background_color = 0xFF181818;
    uint32_t low_color = 0xFF5555FF;
    uint32_t high_color = 0x00FFFF55;

    olivec_fill(img, background_color);

    int neuron_radius = 20;
    int layer_border_vpad = 50;
    int layer_border_hpad = 50;

    int nn_width = img.width - 2*layer_border_hpad;
    int nn_height = img.height - 2*layer_border_vpad;
    int nn_x = img.width/2 - nn_width/2;
    int nn_y = img.height/2 - nn_height/2;
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
                    uint32_t alpha = floorf(255.f*sigmoidf(MAT_AT(nn.ws[l], j, i)));
                    uint32_t connection_color = low_color;
                    olivec_blend_color(&connection_color, (alpha<<(8*3))|high_color);
                    olivec_line(img, cx1, cy1, cx2, cy2, connection_color);
                }
            }
            if (l > 0) {
                uint32_t alpha = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
                uint32_t neuron_color = low_color;
                olivec_blend_color(&neuron_color, (alpha<<(8*3))|high_color);
                olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);
            } else {
                olivec_circle(img, cx1, cy1, neuron_radius, neuron_color_inactivated);
            }
        }
    }
}

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

    NN_PRINT(nn);

    float rate = 1;

    printf("cost = %f\n", nn_cost(nn, ti, to));
    for (size_t i = 0; i < 5*1000; ++i) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        if (i%100 == 0) {
            printf("%zu: cost = %f\n", i + 1, nn_cost(nn, ti, to));
            Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);
            nn_render(img, nn);
            uint32_t frame_thick = 10;
            uint32_t frame_color = 0xFFaaaaaa;
            olivec_frame(img, 0, 0, img.width-1, img.height-1, frame_thick, frame_color);
            char img_file_path[256];
            snprintf(img_file_path, sizeof(img_file_path), "out/xor/xor-%04zu.png", i);
            if (!stbi_write_png(img_file_path, img.width, img.height, 4, img.pixels, img.stride*sizeof(uint32_t))) {
                printf("ERROR: could not save file %s\n", img_file_path);
                return 1;
            }
            printf("INFO : saved NN to %s\n", img_file_path);
        }
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
