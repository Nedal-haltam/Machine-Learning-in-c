

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
#include "NN.h"




std::vector<float> cost = {};
int w = 800;
int h = 600;
NN::NN nn;

float r = 9;
float pad = 5;

void nn_render(NN::NN nn, raylib::Rectangle boundary) {
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
        nn_rand(nn);
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

NN::ModelInput Adder(int BITS)
{
    size_t n = (static_cast<size_t>(1) << BITS);
    size_t rows = n * n;

    std::vector<size_t> NNstruct = std::vector<size_t>();
    NNstruct.push_back(2 * BITS);
    NNstruct.push_back(4 * BITS);
    NNstruct.push_back(BITS + 1);
    NN::ModelInput MI =
    {
        NN::mat_alloc(NULL, rows, 2 * BITS),
        NN::mat_alloc(NULL, rows, BITS + 1),
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

NN::ModelInput XorGate()
{
    std::vector<size_t> NNstruct = std::vector<size_t>();
    NNstruct.push_back(2);
    NNstruct.push_back(2);
    NNstruct.push_back(1);
    NN::ModelInput MI =
    {
        NN::mat_alloc(NULL, 4, 2),
        NN::mat_alloc(NULL, 4, 1),
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

int main(void) {

    NN::Arena arena = NN::arena_alloc_alloc((size_t)16 * 1024 * 1024);
    NN::Arena* arenaloc = &arena;
    size_t mini_batch_size = 1;
    float RegParam = 0;
    float LearRate = 0.1f;
    size_t epochs = 1;

    // NN::ModelInput MI = XorGate();
    NN::ModelInput MI = Adder(6);
    nn = nn_alloc(NULL, MI);
    nn_rand(nn);

    raylib::SetRandomSeed((unsigned int)time(0));
    raylib::SetConfigFlags(raylib::FLAG_WINDOW_RESIZABLE | raylib::FLAG_WINDOW_ALWAYS_RUN);
    raylib::SetTargetFPS(0);
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

        nn_render(nn, NNboundary);
        learn(arenaloc, nn, MI.ti, MI.to, epochs, mini_batch_size, LearRate, RegParam);
        float c = nn_cost(nn, MI.ti, MI.to);
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



