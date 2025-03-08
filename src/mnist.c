//#include "stdio.h"
//#include "stdlib.h"
//#include "time.h"
//#include "assert.h"
//#include "math.h"
//#include "mnistfile.h"
//#include "NN.h"
//
//#define train_cap 256 * 1024 * 1024
//
//static void eval_mnist(NN nn, Mat testinput, Mat testoutput) {
//    size_t eval = 0;
//    for (size_t i = 0; i < testinput.rows; i++) {
//        mat_copy(NN_INPUT(nn), mat_row(testinput, i));
//        feed_forward(nn);
//        size_t pred = mat_max_ind_inrow(NN_OUTPUT(nn), 0);
//        size_t output = mat_max_ind_inrow(mat_row(testoutput, i), 0);
//        if (pred == output) {
//            eval += 1;
//        }
//    }
//    printf("Evaluation of test data : %zu / %zu\n\n", eval, testinput.rows);
//}
//
//
//int main(void) {
//
//    srand(time(0));
//
//    clock_t start, end;
//    double cpu_time_used;
//
//    start = clock();
//
//    intermed = arena_alloc_alloc(256 * 1024 * 1024);
//    Arena* intermedarenaloc = &intermed;
//
//    size_t nn_struct[] = { 784, 30, 10 };
//    size_t mini_batch_size = 10;
//    float RegParam = 5;
//    float LearRate = 0.5f;
//    size_t epochs = 10;
//
//    NN nn = nn_alloc(NULL, nn_struct, ARRAY_LEN(nn_struct));
//    nn_rand(nn);
//
//
//    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
//    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);
//    //for (size_t i = 0; i < 1; i++) {
//    //    for (size_t j = 0; j < 784; j++) {
//    //        printf("pixel[%3zu] = %u \n", j, train_dataset->images[i].pixels[j]);
//
//    //    }
//    //    printf("label[%3zu] = %u \n", i, train_dataset->labels[i]);
//    //}
//    size_t Numofinputs = 784;
//    size_t Numofoutputs = 10;
//    size_t trainsize = 50 * 1000;
//    size_t testsize = 10 * 1000;
//    Mat traininput = trainset_to_mat(intermedarenaloc, Numofinputs, 0, trainsize);
//    Mat trainoutput = trainset_to_mat(intermedarenaloc, 0, Numofoutputs, trainsize);
//    Mat testinput = testset_to_mat(intermedarenaloc, Numofinputs, 0, testsize);
//    Mat testoutput = testset_to_mat(intermedarenaloc, 0, Numofoutputs, testsize);
//    printf("intermed : %zu\n", arena_occupied_bytes(intermedarenaloc));
//    learn(intermedarenaloc, nn, traininput, trainoutput, epochs, mini_batch_size, LearRate, RegParam);
//
//
//    eval_mnist(nn, testinput, testoutput);
//    end = clock();
//    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//    printf("\n\n time taken : %lf\n\n", cpu_time_used);
//    return 0;
//}
