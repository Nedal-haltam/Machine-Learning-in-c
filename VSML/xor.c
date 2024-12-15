//#include "stdio.h"
//#include "stdlib.h"
//#include "time.h"
//#include "assert.h"
//#include "math.h"
//#include "mnistfile.h"
//#include "NN.h"
//
//static void evaluate_gate(NN nn, Mat testinput, Mat testoutput) {
//    size_t eval = 0;
//    for (size_t i = 0; i < testinput.rows; i++) {
//        mat_copy(NN_INPUT(nn), mat_row(testinput, i));
//        feed_forward(nn);
//        size_t pred = mat_max_ind_inrow(NN_OUTPUT(nn), 0);
//        size_t output = mat_max_ind_inrow(mat_row(testoutput, i), 0);
//        if (testoutput.cols == 1) {
//            if (fabsf(MAT_AT(testoutput, i, 0) - MAT_AT(nn.layers[nn.count - 1].a, 0, 0)) < 0.1) {
//                eval += 1;
//            }
//        }
//        else {
//            if (pred == output) {
//                eval += 1;
//            }
//        }
//    }
//    printf("Evaluation of test data : %zu / %zu\n\n", eval, testinput.rows);
//
//    Mat a0 = mat_alloc(NULL, 1, NN_INPUT(nn).cols);
//    
//    for (size_t i = 0; i < 2; i++) {
//        for (size_t j = 0; j < 2; j++) {
//            MAT_AT(a0, 0, 0) = (float)i;
//            MAT_AT(a0, 0, 1) = (float)j;
//            mat_copy(NN_INPUT(nn), a0);
//            feed_forward(nn);
//            float y = MAT_AT(nn.layers[nn.count - 1].a, 0, 0);
//            printf("%zu ^ %zu = %f \n", i, j, y);
//
//        }
//    }
//    free(a0.es);
//}
//
//
//
//int main(void) {
//    srand(time(0));
//
//    clock_t start, end;
//    double cpu_time_used;
//
//    start = clock();
//    size_t nn_struct[] = { 2, 2, 1 };
//    size_t mini_batch_size = 1;
//    float RegParam = 0;
//    float LearRate = 0.5;
//    size_t epochs = 10000;
//    intermed = arena_alloc_alloc(256 * 1024 * 1024);
//    Arena* intermedarenaloc = &intermed;
//    NN nn = nn_alloc(NULL, nn_struct, ARRAY_LEN(nn_struct));
//    nn_rand(nn);
//
//
//    Mat traininput  = mat_alloc(intermedarenaloc, 4, 2);
//    Mat trainoutput = mat_alloc(intermedarenaloc, 4, 1);
//    for (size_t j = 0; j < 2; j++) {
//        for (size_t k = 0; k < 2; k++) {
//            size_t row = 2 * j + k;
//            MAT_AT(traininput, row, 0) = j;
//            MAT_AT(traininput, row, 1) = k;
//            MAT_AT(trainoutput, row, 0) = j ^ k;
//        }
//    }
//
//    Mat testinput = traininput;
//    Mat testoutput = trainoutput;
//    learn(intermedarenaloc, nn, traininput, trainoutput, epochs, mini_batch_size, LearRate, RegParam);
//    evaluate_gate(nn, testinput, testoutput);
//    end = clock();
//    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//    printf("\n\n time taken : %lf\n\n", cpu_time_used);
//    return 0;
//}
//
//
//// draw EM wave using raylib...
////#define size 60.0f
////int w = 800;
////int h = 600;
////float vals[(size_t) size] = { 0 };
////void updatevals(float limit, float pad) {
////    for (size_t i = 0; i < size; i++) {
////        float val = 0.5f * (sinf(2 * PI * vals[i]));
////        Rectangle rec = {
////            .x = i * (w / size) + pad,
////            .y = limit - limit * val,
////            .width = w / size - pad,
////            .height = limit * fabsf(val),
////        };
////        if (val < 0)
////            rec.y = h - (limit + val);
////        DrawRectangleRec(rec, RED);
////
////        rec.y = limit + limit * val;
////        if (val > 0)
////            rec.y = h - (limit + val);
////        DrawRectangleRec(rec, BLUE);
////
////        vals[i] += eps;
////    }
////}
////SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_ALWAYS_RUN);
////InitWindow(w, h, "NN");
////SetTargetFPS(120);
////float pad = 5;
////float limit = h/2;
////for (size_t i = 0; i < size; i++) {
////    vals[i] = (float)i / (size - 1);
////}
////while (!WindowShouldClose()) {
////    w = GetScreenWidth();
////    h = GetScreenHeight();
////    BeginDrawing();
////    ClearBackground(BLACK);
////    updatevals(limit, pad);
////    
////    EndDrawing();
////}
////
////CloseWindow();
////return 0;
//
