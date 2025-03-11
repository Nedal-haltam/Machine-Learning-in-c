#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "float.h"
#include "assert.h"
#include "math.h"
#include "raylib.h"
#include "NN.h"
#include "process.h"
#define nob_da_append_size_t(da, item) \
    do {                                                                             \
        if ((da)->count >= (da)->capacity) {                                         \
            (da)->capacity = (da)->capacity == 0 ? 8 : (da)->capacity*2;             \
            (da)->items = (size_t*)realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                       \
        }                                                                            \
        (da)->items[(da)->count++] = (item);                                         \
    } while (0)
#define nob_da_append_float(da, item) \
    do {                                                                             \
        if ((da)->count >= (da)->capacity) {                                         \
            (da)->capacity = (da)->capacity == 0 ? 8 : (da)->capacity*2;             \
            (da)->items = (float*)realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                       \
        }                                                                            \
        (da)->items[(da)->count++] = (item);                                         \
    } while (0)



Array_float cost = { 0 };
int w = 800;
int h = 600;
NN nn;

float r = 9;
float pad = 5;

//Array nn_struct_from_file(const char* file_path) {
//    Array nn_struct = {0};
//    int buffer_size;
//    unsigned char* buffer = LoadFileData(file_path, &buffer_size);
//
//    String_View content = sv_from_parts(buffer, buffer_size);
//
//    content = sv_trim_left(content);
//    while (content.count > 0 && isdigit(content.data[0])) {
//        size_t x = sv_chop_u64(&content);
//        nob_da_append(&nn_struct, (size_t)x);
//        content = sv_trim_left(content);
//    }
//
//    return nn_struct;
//}

void nn_render(NN nn, Rectangle boundary) {
    size_t n = nn.count;
    assert(n > 0);
    float r = 10;
    float layer_gap = boundary.width / n;
    
    for (size_t i = 0; i < nn.layers[0].a.cols; i++) {
        Vector2 center = { 0 };
        center.x = boundary.x + layer_gap / 2;
        center.y = boundary.y + (i + 1) * (boundary.height / (nn.layers[0].a.cols + 1));
        DrawCircleV(center, r, DARKGRAY);
        for (size_t k = 0; k < nn.layers[1].w.cols; k++) {
            Vector2 end = {
                end.x = boundary.x + layer_gap + layer_gap / 2,
                end.y = boundary.y + (k + 1) * (boundary.height / (nn.layers[1].a.cols + 1)),
            };
            DrawLineEx(center, end, 1, WHITE);
        }
    }
    // Drawing the network
    for (size_t l = 1; l < n; l++) {
        Vector2 center = { 0 };
        size_t neuron_gap = (size_t) boundary.height / nn.layers[l].a.cols;
        for (size_t i = 0; i < nn.layers[l].a.cols; i++) {

            unsigned char b = (unsigned char) (255 * MAT_AT(nn.layers[l].b, 0, i));
            unsigned char w = (unsigned char) (255 * MAT_AT(nn.layers[l].w, 0, i));

            Color color = { w, b, 128, 255 };
            center.x = boundary.x + (l)*layer_gap + layer_gap / 2;
            center.y = boundary.y + (i + 1) * (boundary.height / (nn.layers[l].a.cols + 1));
            DrawCircleV(center, r, color);
            for (size_t k = 0; l < n-1 && k < nn.layers[l+1].a.cols; k++) {
                Vector2 end = {
                    end.x = boundary.x + (l+1) * layer_gap + layer_gap / 2,
                    end.y = boundary.y + (k + 1) * (boundary.height / (nn.layers[l+1].a.cols + 1)),
                };
                DrawLineEx(center, end, 1, WHITE);
            }
        }
    }
}

void ToggleFullScreen() {
    if (IsWindowMaximized()) {
        RestoreWindow();
    }
    else {
        MaximizeWindow();
    }
}
void update() {
    w = GetScreenWidth();
    h = GetScreenHeight();
    if (IsKeyDown(KEY_R)) {
        nn_rand(nn);
        cost.Destruct();
    }
    if (IsKeyPressed(KEY_F)) {

        ToggleFullScreen();
    }
}

void cost_max(Array_float cost, float *max) {
    *max = FLT_MIN;
    for (size_t i = 0; i < cost.count; i++) {
        if (*max < cost.items[i]) {
            *max = cost.items[i];
        }
    }
}

void plot_cost(Array_float cost, Rectangle boundary) {
    Vector2 origin = { origin.x = boundary.x, origin.y = boundary.x + boundary.height };
    DrawLineEx({ boundary.x, boundary.y }, origin, 2, BLUE);
    DrawLineEx(origin, { origin.x + boundary.width, origin.y}, 2, BLUE);

    float max;
    cost_max(cost, &max);
    size_t n = cost.count;
    if (n < 100) n = 100;
    for (size_t i = 0; i+1 < cost.count; i++) {
        Vector2 start = {
            start.x = boundary.x + (float)boundary.width / n * i,
            start.y = boundary.y + (1 - (cost.items[i]) / (max)) * boundary.height,
        };
        Vector2 end = {
            end.x = boundary.x + (float)boundary.width / n * (i+1),
            end.y = boundary.y + (1 - (cost.items[i+1]) / (max)) * boundary.height,
        };
        DrawLineEx(start, end, boundary.height*0.002, RED);
    }
}

ModelInput Adder(int BITS)
{
    size_t n = (static_cast<size_t>(1) << BITS);
    size_t rows = n * n;

    Array_size_t NNstruct = {0};
    nob_da_append_size_t(&NNstruct, 2 * BITS);
    nob_da_append_size_t(&NNstruct, 4 * BITS);
    nob_da_append_size_t(&NNstruct, BITS + 1);
    ModelInput MI = 
    {
        mat_alloc(NULL, rows, 2 * BITS),
        mat_alloc(NULL, rows, BITS + 1),
        NNstruct,
    };
    for (size_t i = 0; i < MI.ti.rows; i++) { // for every input in ti
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y; // the sum
        size_t OF = z >= n; // if the sum is larger than the largest value
        for (size_t j = 0; j < BITS; j++) { 
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

ModelInput XorGate()
{
    Array_size_t NNstruct = { 0 };
    nob_da_append_size_t(&NNstruct, 2);
    nob_da_append_size_t(&NNstruct, 2);
    nob_da_append_size_t(&NNstruct, 1);
    ModelInput MI =
    {
        mat_alloc(NULL, 4, 2),
        mat_alloc(NULL, 4, 1),
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

    Arena arena = arena_alloc_alloc((size_t) 16 * 1024 * 1024);
    Arena* arenaloc = &arena;
    size_t mini_batch_size = 1;
    float RegParam = 0;
    float LearRate = 0.1;
    size_t epochs = 1;

    //ModelInput MI = XorGate();
    ModelInput MI = Adder(5);
    //printf("NNstruct[0] = %u", MI.nn_struct[0]);
    //return 0;
    nn = nn_alloc(NULL, MI);
    nn_rand(nn);

    SetRandomSeed((unsigned int)time(0));
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_ALWAYS_RUN);
    InitWindow(w, h, "NN");
    SetTargetFPS(0);
    
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground({ 0x18, 0x18, 0x18, 0x18 });
        DrawFPS(0, 0);
        update();
        float boundw = 0.6f * w;
        float boundh = 0;
        Rectangle NNboundary = {
            boundw,
            boundh,
            w - (boundw),
            h / 2,
        };

        nn_render(nn, NNboundary);
        learn(arenaloc, nn, MI.ti, MI.to, epochs, mini_batch_size, LearRate, RegParam);
        learn(arenaloc, nn, MI.ti, MI.to, epochs, mini_batch_size, LearRate, RegParam);
        float c = nn_cost(nn, MI.ti, MI.to);
        nob_da_append_float(&cost, c);
        Rectangle plot_boundary = {
            30,
            30,
            w - NNboundary.width,
            h-150,
        };
        plot_cost(cost, plot_boundary);

        arena_reset(&arena);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
