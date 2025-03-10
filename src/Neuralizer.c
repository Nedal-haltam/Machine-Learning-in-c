#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "float.h"
#include "assert.h"
#include "math.h"
#include "raylib.h"
#include "NN.h"
#include "process.h"

typedef struct {
    size_t count;
    size_t capacity;
    size_t* items;
} Array_size_t;
typedef struct {
    size_t count;
    size_t capacity;
    float* items;
} Array_float;

Array_float cost = { 0 };

#define nob_da_append(da, item) \
    do {                                                                             \
        if ((da)->count >= (da)->capacity) {                                         \
            (da)->capacity = (da)->capacity == 0 ? 8 : (da)->capacity*2;             \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                       \
        }                                                                            \
        (da)->items[(da)->count++] = (item);                                         \
    } while (0)


int w = 800;
int h = 600;
NN nn;

float r = 9;
float pad = 5;

static void ToggleFullScreen() {
    if (IsWindowMaximized()) {
        RestoreWindow();
    }
    else {
        MaximizeWindow();
    }
}

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
                .x = boundary.x + layer_gap + layer_gap / 2,
                .y = boundary.y + (k + 1) * (boundary.height / (nn.layers[1].a.cols + 1)),
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

            Color color = (Color) { w, b, 128, 255 };
            center.x = boundary.x + (l)*layer_gap + layer_gap / 2;
            center.y = boundary.y + (i + 1) * (boundary.height / (nn.layers[l].a.cols + 1));
            DrawCircleV(center, r, color);
            for (size_t k = 0; l < n-1 && k < nn.layers[l+1].a.cols; k++) {
                Vector2 end = {
                    .x = boundary.x + (l+1) * layer_gap + layer_gap / 2,
                    .y = boundary.y + (k + 1) * (boundary.height / (nn.layers[l+1].a.cols + 1)),
                };
                DrawLineEx(center, end, 1, WHITE);
            }
        }
    }
}

void update() {
    w = GetScreenWidth();
    h = GetScreenHeight();
    if (IsKeyDown(KEY_R)) {
        nn_rand(nn);
        free(cost.items);
        cost.items = NULL;
        cost.count = 0;
        cost.capacity = 0;
    }
    if (IsKeyPressed(KEY_F)) {

        ToggleFullScreen();
    }
    if (IsKeyPressed(KEY_B)) {
        ToggleBorderlessWindowed();
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
    Vector2 origin = { .x = boundary.x, .y = boundary.x + boundary.height };
    DrawLineEx((Vector2) { .x = boundary.x, .y = boundary.y }, origin, 2, BLUE);
    DrawLineEx(origin, (Vector2) {.x = origin.x + boundary.width, .y = origin.y}, 2, BLUE);

    float max;
    cost_max(cost, &max);
    size_t n = cost.count;
    if (n < 100) n = 100;
    for (size_t i = 0; i+1 < cost.count; i++) {
        Vector2 start = {
            .x = boundary.x + (float)boundary.width / n * i,
            .y = boundary.y + (1 - (cost.items[i]) / (max)) * boundary.height,
        };
        Vector2 end = {
            .x = boundary.x + (float)boundary.width / n * (i+1),
            .y = boundary.y + (1 - (cost.items[i+1]) / (max)) * boundary.height,
        };
        DrawLineEx(start, end, boundary.height*0.002, RED);
    }
}

typedef struct ModelInput
{
    Mat ti, to;
    size_t nn_struct[100];
    size_t count;
} ModelInput;

ModelInput Adder(int BITS)
{
    size_t n = (1 << BITS);
    size_t rows = n * n;
    ModelInput MI = 
    {
        .ti = mat_alloc(NULL, rows, 2 * BITS),
        .to = mat_alloc(NULL, rows, BITS + 1),
        .nn_struct = { 2 * BITS, 4 * BITS, BITS + 1 },
        .count = 3,
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
    ModelInput MI =
    {
        .ti = mat_alloc(NULL, 4, 2),
        .to = mat_alloc(NULL, 4, 1),
        .nn_struct = { 2, 2, 1 },
        .count = 3,
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
    nn = nn_alloc(NULL, MI.nn_struct, MI.count);
    nn_rand(nn);

    SetRandomSeed((unsigned int)time(0));
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_ALWAYS_RUN);
    InitWindow(w, h, "NN");
    SetTargetFPS(0);
    
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground((Color) { 0x18, 0x18, 0x18, 0x18 });

        update();
        float boundw = 0.6f * w;
        float boundh = 0;
        Rectangle NNboundary = {
            .x = boundw,
            .y = boundh,
            .width  = w - (boundw),
            .height = h / 2,
        };

        nn_render(nn, NNboundary);
        learn(arenaloc, nn, MI.ti, MI.to, epochs, mini_batch_size, LearRate, RegParam);
        learn(arenaloc, nn, MI.ti, MI.to, epochs, mini_batch_size, LearRate, RegParam);
        float c = nn_cost(nn, MI.ti, MI.to);
        nob_da_append(&cost, c);
        Rectangle plot_boundary = {
            .x = 30,
            .y = 30,
            .width = w - NNboundary.width,
            .height = h-150,
        };
        plot_cost(cost, plot_boundary);

        arena_reset(&arena);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
