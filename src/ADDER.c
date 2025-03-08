//#include "stdio.h"
//#include "stdlib.h"
//#include "time.h"
//#include "assert.h"
//#include "math.h"
//
//#include "NN.h"
//
//
//#define BITS 5
//int main(void) {
//    SetRandomSeed((unsigned int)time(0));
//    clock_t start, end;
//    double cpu_time_used;
//
//    start = clock();
//
//    intermed = arena_alloc_alloc(16 * 1024 * 1024);
//    Arena* intermedarenaloc = &intermed;
//
//    size_t n = (1 << BITS);
//    size_t rows = n * n;
//    Mat ti = mat_alloc(NULL, rows, 2 * BITS);
//    Mat to = mat_alloc(NULL, rows, BITS + 1);
//
//    for (size_t i = 0; i < ti.rows; i++) { // for every input in ti
//        size_t x = i / n;
//        size_t y = i % n;
//        size_t z = x + y; // the sum
//        size_t OF = z >= n; // if the sum is larger than the largest value
//        for (size_t j = 0; j < BITS; j++) { 
//            MAT_AT(ti, i, j) = (x >> j) & 1; // get every bit corresponding to that number
//            MAT_AT(ti, i, j + BITS) = (y >> j) & 1;
//            if (OF) { // if OF then output is zero we don't care
//                MAT_AT(to, i, j) = 0;
//            }
//            else { // else we calculate it per bit
//                MAT_AT(to, i, j) = (z >> j) & 1;
//            }
//        }
//        MAT_AT(to, i, BITS) = OF; // the OF flag
//    }
//
//    //Mat testinput = ti;
//    //Mat testoutput = to;
//    
//    size_t nn_struct[] = { 2 * BITS, 4 * BITS, BITS + 1 };
//    size_t mini_batch_size = 1;
//    float RegParam = 0;
//    //float LearRate = 1; // finite diff with sigmoid or tanh
//    float LearRate = 0.1; // back prop.....
//    size_t epochs = 3000;
//    
//    NN nn = nn_alloc(NULL, nn_struct, ARRAY_LEN(nn_struct));
//    nn_rand(nn);
//    learn(intermedarenaloc, nn, ti, to, epochs, mini_batch_size, LearRate, RegParam);
//
//
//    for (size_t i = 0; i < to.rows; i++) {
//        mat_copy(NN_INPUT(nn), mat_row(ti, i));
//        feed_forward(nn);
//        for (size_t j = 0; j < to.cols; j++) {
//            if (fabsf(MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(mat_row(to, i), 0, j)) > 0.05) {
//                MAT_PRINT(NN_OUTPUT(nn));
//                MAT_PRINT(mat_row(to, i));
//                break;
//            }
//        }
//    }
//
//    end = clock();
//    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//    printf("\n\n time taken : %lf\n\n", cpu_time_used);
//
//    return 0;
//}
