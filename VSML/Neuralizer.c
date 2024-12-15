#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "float.h"
#include "assert.h"
#include "math.h"
#include "raylib.h"
#include "NN.h"
#define SV_IMPLEMENTATION
#include "sv.h"

typedef struct {
    size_t count;
    size_t capacity;
    size_t* items;
} Array;
typedef struct {
    size_t count;
    size_t capacity;
    float* items;
} Array2;
typedef struct {
    Vector2 start;
    Vector2 end;
} point;
typedef struct {
    point* vals;
} points;

Array2 cost = { 0 };

#define nob_da_append(da, item) \
    do {                                                                             \
        if ((da)->count >= (da)->capacity) {                                         \
            (da)->capacity = (da)->capacity == 0 ? 8 : (da)->capacity*2;             \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                       \
        }                                                                            \
        (da)->items[(da)->count++] = (item);                                         \
    } while (0)


//#define size 1000
int w = 800;
int h = 600;
NN nn;

points graph;
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

Array nn_struct_from_file(const char* file_path) {
    Array nn_struct = {0};
    int buffer_size;
    unsigned char* buffer = LoadFileData(file_path, &buffer_size);

    String_View content = sv_from_parts(buffer, buffer_size);

    content = sv_trim_left(content);
    while (content.count > 0 && isdigit(content.data[0])) {
        size_t x = sv_chop_u64(&content);
        nob_da_append(&nn_struct, (size_t)x);
        content = sv_trim_left(content);
    }

    return nn_struct;
}

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
            DrawLineEx(center, end, 1, BLACK);
        }
    }
    // Drawing the network
    for (size_t l = 1; l < n; l++) {
        Vector2 center = { 0 };
        size_t neuron_gap = (size_t) boundary.height / nn.layers[l].a.cols;
        for (size_t i = 0; i < nn.layers[l].a.cols; i++) {

            unsigned char b = (unsigned char) (255 * MAT_AT(nn.layers[l].b, 0, i));
            unsigned char w = (unsigned char) (255 * MAT_AT(nn.layers[l].w, 0, i));

            Color color = CLITERAL(Color) { w, b, w*b, 255 };
            center.x = boundary.x + (l)*layer_gap + layer_gap / 2;
            center.y = boundary.y + (i + 1) * (boundary.height / (nn.layers[l].a.cols + 1));
            DrawCircleV(center, r, color);
            for (size_t k = 0; l < n-1 && k < nn.layers[l+1].a.cols; k++) {
                Vector2 end = {
                    .x = boundary.x + (l+1) * layer_gap + layer_gap / 2,
                    .y = boundary.y + (k + 1) * (boundary.height / (nn.layers[l+1].a.cols + 1)),
                };
                DrawLineEx(center, end, 1, BLACK);
            }
        }
    }
}

void Init() {
    SetRandomSeed((unsigned int)time(0));
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_ALWAYS_RUN);
    InitWindow(w, h, "NN");
    SetTargetFPS(60);
}

void update() {
    ClearBackground(ColorBrightness(GRAY, 0.4f));
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

void cost_max(Array2 cost, float *max) {
    *max = FLT_MIN;
    for (size_t i = 0; i < cost.count; i++) {
        if (*max < cost.items[i]) {
            *max = cost.items[i];
        }
    }
}

void plot_cost(Array2 cost, Rectangle boundary) {
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

#define BITS 5
int main(void) {
    //graph.vals = malloc(sizeof(*graph.vals) * size);
    //float theta = 0;
    Init();
    Arena arena = arena_alloc_alloc((size_t) 16 * 1024 * 1024);
    Arena* arenaloc = &arena;

    //Array nn_struct = nn_struct_from_file("./demo.txt");
    size_t mini_batch_size = 1;
    float RegParam = 0;
    float LearRate = 0.1;
    size_t epochs = 1;
    size_t n = (1 << BITS);
    size_t rows = n * n;
    Mat ti = mat_alloc(NULL, rows, 2 * BITS);
    Mat to = mat_alloc(NULL, rows, BITS + 1);
    size_t nn_struct[] = { 2 * BITS, 4 * BITS, BITS + 1 };

    for (size_t i = 0; i < ti.rows; i++) { // for every input in ti
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y; // the sum
        size_t OF = z >= n; // if the sum is larger than the largest value
        for (size_t j = 0; j < BITS; j++) { 
            MAT_AT(ti, i, j) = (x >> j) & 1; // get every bit corresponding to that number
            MAT_AT(ti, i, j + BITS) = (y >> j) & 1;
            if (OF) { // if OF then output is zero we don't care
                MAT_AT(to, i, j) = 0;
            }
            else { // else we calculate it per bit
                MAT_AT(to, i, j) = (z >> j) & 1;
            }
        }
        MAT_AT(to, i, BITS) = OF; // the OF flag
    }
    
    //Mat traininput  = mat_alloc(arenaloc, 4, 2);
    //Mat trainoutput = mat_alloc(arenaloc, 4, 1);
    //for (size_t j = 0; j < 2; j++) {
    //    for (size_t k = 0; k < 2; k++) {
    //        size_t row = 2 * j + k;
    //        MAT_AT(traininput, row, 0) = j;
    //        MAT_AT(traininput, row, 1) = k;
    //        MAT_AT(trainoutput, row, 0) = j ^ k;
    //    }
    //}
    
    //evaluate_gate(nn, testinput, testoutput);
    nn = nn_alloc(NULL, nn_struct, ARRAY_LEN(nn_struct));
    nn_rand(nn);
    
    while (!WindowShouldClose()) {
        BeginDrawing();
        update();

        float boundw = (float) 0.6*w;
        float boundh = (float) 0;
        Rectangle NNboundary = {
            .x = boundw,
            .y = boundh,
            .width  = w - (boundw),
            .height = h / 2,
        };

        nn_render(nn, NNboundary);
        learn(arenaloc, nn, ti, to, epochs, mini_batch_size, LearRate, RegParam);
        learn(arenaloc, nn, ti, to, epochs, mini_batch_size, LearRate, RegParam);
        float c = nn_cost(nn, ti, to);
        nob_da_append(&cost, 1*c);
        Rectangle plot_boundary = {
            .x = 30,
            .y = 30,
            .width = w - NNboundary.width,
            .height = h-150,
        };
        plot_cost(cost, plot_boundary);
        //DrawRectangleRec(plot_boundary, RED);





        arena_reset(&arena);
        EndDrawing();
/*
        theta += 0.05;
        for (size_t i = 0; i < size; i++) {
            float t = (float)i * ((float)w / (size - 1)) - w / 2;
            //float val = 10 * sinf(2 * PI * t);
            float val = powf(0.25*t, 2);
            if (theta >= 2 * PI) {
                theta = 0;
            }
            
            graph.vals[i].end.x = t;
            graph.vals[i].end.y = -val;
            
            float cosres = cosf(theta);
            float sinres = sinf(theta);

            float oldx = graph.vals[i].end.x;
            float oldy = graph.vals[i].end.y;

            graph.vals[i].end.x = oldx * cosres - oldy * sinres + w / 2;
            graph.vals[i].end.y = oldx * sinres + oldy * cosres + w / 2;
            DrawCircleV(graph.vals[i].end, r, RED);
        }
        */
    }

    CloseWindow();
    return 0;
}
