

#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "float.h"
#include "assert.h"
#include "math.h"
#include <vector>
namespace raylib {
    #include <raylib.h>
}
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "nn.h"

std::vector<float> cost = {};
int w = 800;
int h = 600;
nn::Nnet nnet;

float r = 9;
float pad = 5;

void nn_render(nn::Nnet nn, raylib::Rectangle boundary) {
    size_t n = nn.count;
    assert(n > 0);
    float r = 10;
    float layer_gap = boundary.width / n;

    for (size_t i = 0; i < nn.layers[0].a.cols; i++) {
        raylib::Vector2 center = {};
        center.x = boundary.x + layer_gap / 2;
        center.y = boundary.y + (i + 1) * (boundary.height / (nn.layers[0].a.cols + 1));
        raylib::DrawCircleV(center, r, raylib::DARKGRAY);
        for (size_t k = 0; k < nn.layers[1].w.cols; k++) {
            raylib::Vector2 end = {
                end.x = boundary.x + layer_gap + layer_gap / 2,
                end.y = boundary.y + (k + 1) * (boundary.height / (nn.layers[1].a.cols + 1)),
            };
            raylib::DrawLineEx(center, end, 1, raylib::WHITE);
        }
    }
    // Drawing the network
    for (size_t l = 1; l < n; l++) {
        raylib::Vector2 center = raylib::Vector2{};
        size_t neuron_gap = (size_t)boundary.height / nn.layers[l].a.cols;
        (void)neuron_gap;
        for (size_t i = 0; i < nn.layers[l].a.cols; i++) {

            unsigned char b = (unsigned char)(255 * MAT_AT(nn.layers[l].b, 0, i));
            unsigned char w = (unsigned char)(255 * MAT_AT(nn.layers[l].w, 0, i));

            raylib::Color color = { w, b, 128, 255 };
            center.x = boundary.x + (l)*layer_gap + layer_gap / 2;
            center.y = boundary.y + (i + 1) * (boundary.height / (nn.layers[l].a.cols + 1));
            DrawCircleV(center, r, color);
            for (size_t k = 0; l < n - 1 && k < nn.layers[l + 1].a.cols; k++) {
                raylib::Vector2 end = {
                    end.x = boundary.x + (l + 1) * layer_gap + layer_gap / 2,
                    end.y = boundary.y + (k + 1) * (boundary.height / (nn.layers[l + 1].a.cols + 1)),
                };
                DrawLineEx(center, end, 1, raylib::WHITE);
            }
        }
    }
}

void ToggleFullScreen() {
    if (raylib::IsWindowMaximized()) {
        raylib::RestoreWindow();
    }
    else {
        raylib::MaximizeWindow();
    }
}

void update() {
    w = raylib::GetScreenWidth();
    h = raylib::GetScreenHeight();
    if (raylib::IsKeyDown(raylib::KEY_R)) {
        nn_rand(nnet);
        cost.clear();
    }
    if (raylib::IsKeyPressed(raylib::KEY_F)) {

        ToggleFullScreen();
    }
}

void cost_max(std::vector<float> cost, float* max) {
    *max = FLT_MIN;
    for (size_t i = 0; i < cost.size(); i++) {
        if (*max < cost[i]) {
            *max = cost[i];
        }
    }
}

void plot_cost(std::vector<float> cost, raylib::Rectangle boundary) {
    raylib::Vector2 origin = { origin.x = boundary.x, origin.y = boundary.x + boundary.height };
    DrawLineEx({ boundary.x, boundary.y }, origin, 2, raylib::BLUE);
    DrawLineEx(origin, { origin.x + boundary.width, origin.y }, 2, raylib::BLUE);

    float max;
    cost_max(cost, &max);
    size_t n = cost.size();
    if (n < 100) n = 100;
    for (size_t i = 0; i + 1 < cost.size(); i++) {
        raylib::Vector2 start = {
            start.x = boundary.x + (float)boundary.width / n * i,
            start.y = boundary.y + (1 - (cost[i]) / (max)) * boundary.height,
        };
        raylib::Vector2 end = {
            end.x = boundary.x + (float)boundary.width / n * (i + 1),
            end.y = boundary.y + (1 - (cost[i + 1]) / (max)) * boundary.height,
        };
        raylib::DrawLineEx(start, end, boundary.height * 0.002, raylib::RED);
    }
}

nn::ModelInput Adder(int BITS)
{
    size_t n = (static_cast<size_t>(1) << BITS);
    size_t rows = n * n;

    std::vector<size_t> NNstruct = std::vector<size_t>();
    NNstruct.push_back(2 * BITS);
    NNstruct.push_back(4 * BITS);
    NNstruct.push_back(BITS + 1);
    nn::ModelInput MI =
    {
        nn::mat_alloc(NULL, rows, 2 * BITS),
        nn::mat_alloc(NULL, rows, BITS + 1),
        NNstruct,
    };
    for (size_t i = 0; i < MI.ti.rows; i++) { // for every input in ti
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y; // the sum
        size_t OF = z >= n; // if the sum is larger than the largest value
        for (int j = 0; j < BITS; j++) {
            MAT_AT(MI.ti, i, j) = (x >> j) & 1; // get every bit corresponding to that number
            MAT_AT(MI.ti, i, j + BITS) = (y >> j) & 1;
            if (OF) { // if OF then output is zero we don't care
                MAT_AT(MI.to, i, j) = 0;
            }
            else { // else we calculate it per bit
                MAT_AT(MI.to, i, j) = (z >> j) & 1;
            }
        }
        MAT_AT(MI.to, i, BITS) = OF; // the OF flag
    }
    return MI;
}

nn::ModelInput XorGate()
{
    std::vector<size_t> NNstruct = std::vector<size_t>();
    NNstruct.push_back(2);
    NNstruct.push_back(2);
    NNstruct.push_back(1);
    nn::ModelInput MI =
    {
        nn::mat_alloc(NULL, 4, 2),
        nn::mat_alloc(NULL, 4, 1),
        NNstruct,
    };
    for (size_t j = 0; j < 2; j++) {
        for (size_t k = 0; k < 2; k++) {
            size_t row = 2 * j + k;
            MAT_AT(MI.ti, row, 0) = j;
            MAT_AT(MI.ti, row, 1) = k;
            MAT_AT(MI.to, row, 0) = j ^ k;
        }
    }
    return MI;
}

typedef struct {
    int width;
    int height;
    int max_val;
    unsigned char* data;
} PPMImage;

int write_ppm(const char* filename, PPMImage* img) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Cannot write to file");
        return 0;
    }
    fprintf(fp, "P6\n%d %d\n%d\n", img->width, img->height, img->max_val);
    fwrite(img->data, 1, img->width * img->height * 3, fp);
    fclose(fp);
    return 1;
}

PPMImage MIToPPM(nn::Mat ti, nn::Mat to, int width, int height)
{
    PPMImage img = {
        img.width = width,
        img.height = height,
        img.max_val = 255,
    };
    img.data = (unsigned char*)malloc(img.width * img.height * to.cols),
    assert(img.data != NULL);
    for (int i = 0; i < width * height; i++)
    {
        for (int c = 0; c < (int)to.cols; c++)
        {
            unsigned char val = (unsigned char)(MAT_AT(to, i, c) * 255.0f);
            img.data[i * to.cols + c] = val;
        }
    }
    return img;
}

#define L 12
#define P_ENCODING_SIZE (2 * L * 2)
nn::ModelInput Image(const char* filepath)
{
    int x,y,n;
    unsigned char *image = stbi_load(filepath, &x, &y, &n, 0);
    if (!image) {
        printf("Failed to load image\n");
        exit(1);
    }
    printf("Loaded image : %dx%d and %d channels\n", x, y, n);

    // nn::ModelInput mi = {
    //     nn::mat_alloc(NULL, x * y, 2),
    //     nn::mat_alloc(NULL, x * y, n),
    //     std::vector<size_t>{2, 128, 128, 128, n},
    // };
    nn::ModelInput mi = {
        nn::mat_alloc(NULL, x * y, P_ENCODING_SIZE), 
        nn::mat_alloc(NULL, x * y, n),
        std::vector<size_t>{P_ENCODING_SIZE, 256, 256, 256, 256, (size_t)n},
    };


    for (int i = 0; i < x * y; i++)
    {
        int ix = i % x;
        int iy = i / x;
        // MAT_AT(mi.ti, i, 0) = (float)ix / (float)x;
        // MAT_AT(mi.ti, i, 1) = (float)iy / (float)y;

        float nx = (float)ix / (float)x;
        float ny = (float)iy / (float)y;
        int current_col = 0;
        for (int j = 0; j < L; j++)
        {
            float freq = (float)pow(2.0f, j);
            float val = M_PI * freq * nx;
            MAT_AT(mi.ti, i, current_col++) = sinf(val); 
            MAT_AT(mi.ti, i, current_col++) = cosf(val);
        }
        for (int j = 0; j < L; j++)
        {
            float freq = (float)pow(2.0f, j);
            float val = M_PI * freq * ny;
            MAT_AT(mi.ti, i, current_col++) = sinf(val);
            MAT_AT(mi.ti, i, current_col++) = cosf(val);
        }

        for (int c = 0; c < n; c++)
        {
            MAT_AT(mi.to, i, c) = (float)image[i * n + c] / 255.0f;
        }
    }

    // PPMImage ppm = MIToPPM(mi.ti, mi.to, x, y);
    // write_ppm("input.ppm", &ppm);
    // free(ppm.data);
    // printf("Saved input.ppm\n");

    stbi_image_free(image);
    return mi;
}

void SaveNNAsImage(nn::Nnet nn, const char* inputfilepath, const char* outputfilepath)
{
    int x,y,n;
    unsigned char *image = stbi_load(inputfilepath, &x, &y, &n, 0);
    if (!image) {
        printf("Failed to load image\n");
        exit(1);
    }
    stbi_image_free(image);

    nn::ModelInput mi = {
        nn::mat_alloc(NULL, x * y, 2),
        nn::mat_alloc(NULL, x * y, nn.layers[nn.count - 1].a.cols),
    };
    
    nn::Mat a0 = nn::mat_alloc(NULL, 1, NN_INPUT(nn).cols);
    for (size_t i = 0; i < mi.ti.rows; i++) {
        for (size_t j = 0; j < mi.ti.cols; j++) {
            MAT_AT(a0, 0, j) = MAT_AT(mi.ti, i, j);
        }
        mat_copy(NN_INPUT(nn), a0);
        feed_forward(nn);
        for (size_t j = 0; j < mi.to.cols; j++) {
            MAT_AT(mi.to, i, j) = MAT_AT(nn.layers[nn.count - 1].a, 0, j);
        }
    }

    PPMImage ppm = MIToPPM(mi.ti, mi.to, x, y);
    write_ppm(outputfilepath, &ppm);
    free(ppm.data);
    printf("Saved %s\n", outputfilepath);
}

int main(void)
{
    nn::Arena arena = nn::arena_alloc_alloc((size_t)16 * 1024 * 1024);
    nn::Arena* arenaloc = &arena;
    size_t mini_batch_size = 20;
    float RegParam = 0.0001f;
    float LearRate = 0.001f;
    size_t epochs = 500;

    // nn::ModelInput MI = XorGate();
    // nn::ModelInput MI = Adder(6);
    const char* inputfilepath = "./images/lena.png";
    const char* outputfilepath = "./images/output.ppm";
    nn::ModelInput MI = Image(inputfilepath);
    nnet = nn_alloc(NULL, MI);
    srand((unsigned int)time(0));
    nn_rand(nnet);

    raylib::SetRandomSeed((unsigned int)time(0));
    raylib::SetConfigFlags(raylib::FLAG_WINDOW_RESIZABLE | raylib::FLAG_WINDOW_ALWAYS_RUN);
    raylib::SetTargetFPS(0);

    learn(arenaloc, nnet, MI.ti, MI.to, epochs, mini_batch_size, LearRate, RegParam);
    arena_reset(&arena);

    SaveNNAsImage(nnet, inputfilepath, outputfilepath);

    return 0;

    raylib::InitWindow(w, h, "NN");
    
    while (!raylib::WindowShouldClose()) {
        raylib::BeginDrawing();
        raylib::ClearBackground({ 0x18, 0x18, 0x18, 0x18 });
        raylib::DrawFPS(0, 0);
        update();
        float boundw = 0.6f * w;
        float boundh = 0;
        raylib::Rectangle NNboundary = {
            boundw,
            boundh,
            (float)w - (boundw),
            (float)h / 2,
        };

        nn_render(nnet, NNboundary);
        learn(arenaloc, nnet, MI.ti, MI.to, epochs, mini_batch_size, LearRate, RegParam);
        float c = nn_cost(nnet, MI.ti, MI.to);
        cost.push_back(c);
        raylib::Rectangle plot_boundary = {
            (float)30,
            (float)30,
            (float)w - NNboundary.width,
            (float)h - 150,
        };
        plot_cost(cost, plot_boundary);

        arena_reset(&arena);
        raylib::EndDrawing();
    }

    raylib::CloseWindow();
    return 0;
}



