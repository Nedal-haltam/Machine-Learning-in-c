#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "float.h"
#include "assert.h"
#include "math.h"
#include "process.h"


#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]
#define MIN(a,b) (((a)<(b)) ? (a) : (b))
#define MAX(a,b) (((a)>(b)) ? (a) : (b))
#define ARRAY_LEN(arr) ((sizeof(arr)) / (sizeof((arr)[0])))
#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)
    //#define arena_reset(arena) if (arena) (arena)->size = 0
    //#define arena_occupied_bytes(arena) ((arena) ? (arena)->size * sizeof(*(arena)->data) : 0)
#define arena_reset(arena) ((arena)->size = 0)
#define arena_occupied_bytes(arena) (assert(arena != NULL), (arena)->size * sizeof(*(arena)->data))
#define arena_type char
#define MAT_PRINT(m) mat_print(m, #m)
#define LAY_PRINT(l) lay_print(l, #l)
#define NN_PRINT(nn) nn_print(nn, #nn)
#define NN_OUTPUT(nn) nn.layers[nn.count - 1].a
#define NN_INPUT(nn) (nn).layers[0].a

#define BP 1

namespace NN
{
    typedef enum MatType
    {
        weights, biases
    } MatType;

    typedef struct {
        size_t rows;
        size_t cols;
        size_t stride;
        float* es;
    } Mat;

    typedef struct ModelInput
    {
    public:
        Mat ti, to;
        std::vector<size_t> NNstruct;
    } ModelInput;

    typedef struct {
        Mat w;
        Mat b;
        Mat z;
        Mat a;
    } Layer;

    typedef struct {
        Layer* layers;
        size_t count;
    } NN;

    typedef struct {
        //Arena* next;
        size_t size;
        size_t capacity;
        arena_type* data;
    } Arena;

    static Mat* nabla_b = {};
    static Mat* nabla_w = {};
    static float eps = 0.001f;
    Arena arena_alloc_alloc(size_t capacity_bytes) {
        Arena arena = {};

        // size per word    
        size_t word_size = sizeof(*arena.data);
        // number of words to be allocated
        size_t capacity_words = (capacity_bytes + word_size - 1) / word_size;

        void* data = malloc(capacity_words * word_size);
        assert(data != NULL);

        arena.capacity = capacity_words;
        arena.size = 0;
        arena.data = (arena_type*)data;
        return arena;
    }

    void* arena_alloc(Arena* arena, size_t size_bytes) {

        if (arena == NULL) return malloc(size_bytes);

        // size per word    
        size_t word_size = sizeof(*arena->data);
        // number of words to be allocated
        size_t size_words = (size_bytes + word_size - 1) / word_size;

        // the result is a pointer that points to the first free not allocated word in the arena
        assert(arena->size + size_words <= arena->capacity);
        void* result = &arena->data[arena->size];
        arena->size += size_words;
        return result;
    }


    float rand_float() {
        return (float)rand() / (float)RAND_MAX;
    }

    void mat_fill(Mat m, float x) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) = x;
            }
        }
    }

    Mat mat_alloc(Arena* arena, size_t rows, size_t cols) {
        Mat m = {};
        m.rows = rows;
        m.cols = cols;
        m.stride = cols;
        m.es = (float*)arena_alloc(arena, rows * cols * sizeof(*m.es));
        assert(m.es != NULL);
        mat_fill(m, 0.0f);;
        return m;
    }


    NN nn_alloc(Arena* arena, ModelInput MI) {
        NN nn = {};
        nn.count = MI.NNstruct.size();
        nn.layers = (Layer*)arena_alloc(arena, nn.count * sizeof(*nn.layers));

        for (size_t i = 1; i < nn.count; i++) {
            nn.layers[i].w = mat_alloc(arena, MI.NNstruct[i - 1], MI.NNstruct[i]);
            nn.layers[i].b = mat_alloc(arena, 1, MI.NNstruct[i]);
            nn.layers[i].a = mat_alloc(arena, 1, MI.NNstruct[i]);
            nn.layers[i].z = mat_alloc(arena, 1, MI.NNstruct[i]);
        }
        nn.layers[0].a = mat_alloc(arena, 1, MI.NNstruct[0]);
        return nn;
    }

    void mat_copy(Mat dst, Mat src) {
        assert(dst.rows == src.rows);
        assert(dst.cols == src.cols);

        for (size_t i = 0; i < dst.rows; i++) {
            for (size_t j = 0; j < dst.cols; j++) {
                // float act = MAT_AT(src, i, j);
                MAT_AT(dst, i, j) = MAT_AT(src, i, j);
            }
        }
    }

    Mat mat_row(Mat m, size_t r) {
        Mat ret = {};
        ret.rows = 1;
        ret.cols = m.cols;
        ret.stride = m.stride;
        ret.es = &MAT_AT(m, r, 0);
        return ret;
    }

    Mat mat_mat(Mat m, size_t sr, size_t er, size_t sc, size_t ec) {
        Mat ret = {};
        ret.rows = er - sr + 1;
        ret.cols = ec - sc + 1;
        ret.stride = m.stride;
        ret.es = &MAT_AT(m, sr, sc);
        return ret;
    }

    void mat_transpose(Mat dst, Mat src) {
        assert(dst.rows == src.cols);
        assert(dst.cols == src.rows);
        for (size_t i = 0; i < dst.rows; i++) {
            for (size_t j = 0; j < dst.cols; j++) {
                MAT_AT(dst, i, j) = MAT_AT(src, j, i);
            }
        }
    }

    void mat_rand(Mat m, float lo, float hi) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) = rand_float() * (hi - lo) + lo;
            }
        }
    }

    void mat_incr(Mat m) {
        float temp = 1;
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) = temp;
                temp += 1;
            }
        }
    }

    void mat_dot(Mat c, Mat a, Mat b) {
        assert(c.rows == a.rows);
        assert(c.cols == b.cols);
        assert(a.cols == b.rows);
        for (size_t i = 0; i < c.rows; i++) {
            for (size_t j = 0; j < c.cols; j++) {
                float temp = 0;
                for (size_t k = 0; k < a.cols; k++) {
                    temp += MAT_AT(a, i, k) * MAT_AT(b, k, j);
                }
                MAT_AT(c, i, j) = temp;
            }
        }
    }

    void mat_addEW(Mat c, Mat a, Mat b) {
        assert(c.rows == a.rows && a.rows == b.rows);
        assert(c.cols == a.cols && a.cols == b.cols);
        for (size_t i = 0; i < c.rows; i++) {
            for (size_t j = 0; j < c.cols; j++) {
                float temp = MAT_AT(a, i, j) + MAT_AT(b, i, j);
                MAT_AT(c, i, j) = temp;
            }
        }
    }

    void mat_subEW(Mat c, Mat a, Mat b) {
        assert(c.rows == a.rows && a.rows == b.rows);
        assert(c.cols == a.cols && a.cols == b.cols);
        for (size_t i = 0; i < c.rows; i++) {
            for (size_t j = 0; j < c.cols; j++) {
                float temp = MAT_AT(a, i, j) - MAT_AT(b, i, j);
                MAT_AT(c, i, j) = temp;
            }
        }
    }

    void mat_mulEW(Mat c, Mat a, Mat b) {
        assert(c.rows == a.rows && a.rows == b.rows);
        assert(c.cols == a.cols && a.cols == b.cols);
        for (size_t i = 0; i < c.rows; i++) {
            for (size_t j = 0; j < c.cols; j++) {
                MAT_AT(c, i, j) = MAT_AT(a, i, j) * MAT_AT(b, i, j);
            }
        }
    }

    void mat_divEW(Mat c, Mat a, Mat b) {
        assert(c.rows == a.rows && a.rows == b.rows);
        assert(c.cols == a.cols && a.cols == b.cols);
        for (size_t i = 0; i < c.rows; i++) {
            for (size_t j = 0; j < c.cols; j++) {
                if (MAT_AT(b, i, j) == 0)
                    continue;
                MAT_AT(c, i, j) = MAT_AT(a, i, j) / MAT_AT(b, i, j);
            }
        }
    }

    void mat_exp(Mat m) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) = expf(MAT_AT(m, i, j));
            }
        }
    }

    void mat_pow(Mat m, float x) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) = (float)pow(MAT_AT(m, i, j), x);
            }
        }
    }

    void mat_sqrt(Mat m) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) = sqrtf(MAT_AT(m, i, j));
            }
        }
    }

    void mat_log(Mat m) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                if (MAT_AT(m, i, j) < 1e-7f)
                    MAT_AT(m, i, j) = 1e-7f;
                MAT_AT(m, i, j) = logf(MAT_AT(m, i, j));
            }
        }
    }

    void mat_clip(Mat m, float lo, float hi) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                if (MAT_AT(m, i, j) < lo)
                    MAT_AT(m, i, j) = lo;
                if (MAT_AT(m, i, j) > hi)
                    MAT_AT(m, i, j) = hi;
            }
        }
    }

    void mat_mul_const(Mat m, float x) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) *= x;
            }
        }
    }

    void mat_add_const(Mat m, float x) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) += x;
            }
        }
    }

    void mat_reciprocal(Mat m) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                if (MAT_AT(m, i, j) == 0)
                    continue;
                MAT_AT(m, i, j) = 1.0f / MAT_AT(m, i, j);
            }
        }
    }

    float mat_max(Mat m) {
        float max = MAT_AT(m, 0, 0);
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                if (MAT_AT(m, i, j) > max)
                    max = MAT_AT(m, i, j);
            }
        }
        return max;
    }

    size_t mat_max_ind_incol(Mat m, size_t c) {
        float max = MAT_AT(m, 0, c);
        size_t ind = 0;
        for (size_t i = 0; i < m.rows; i++) {
            if (MAT_AT(m, i, 0) > max) {
                max = MAT_AT(m, i, c);
                ind = i;
            }
        }
        return ind;
    }

    size_t mat_max_ind_inrow(Mat m, size_t r) {
        float max = MAT_AT(m, r, 0);
        size_t ind = 0;
        for (size_t i = 0; i < m.cols; i++) {
            if (MAT_AT(m, 0, i) > max) {
                max = MAT_AT(m, r, i);
                ind = i;
            }
        }
        return ind;
    }

    float mat_sumof(Mat m) {
        float temp = 0;
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                temp += MAT_AT(m, i, j);
            }
        }
        return temp;
    }

    void mat_print(Mat m, const char* name) {

        printf("        %s = [\n", name);
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                printf("            %f ", MAT_AT(m, i, j));
            }
            printf("\n");
        }
        printf("]\n");
    }

    void lay_print(Layer l, const char* name) {
        printf("    %s = [\n", name);
        MAT_PRINT(l.w);
        MAT_PRINT(l.b);
        MAT_PRINT(l.a);
        printf("]\n");
    }

    void nn_print(NN nn, const char* name) {
        printf("%s = [\n", name);
        for (size_t i = 1; i < nn.count; i++) {
            LAY_PRINT(nn.layers[i]);
        }
        printf("]\n");
    }

    float sigmoid(float z) {
        float act = 1.0f / (1.0f + expf(-z));
        return act;
    }

    void mat_sigmoid(Mat m) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                float act = MAT_AT(m, i, j);
                MAT_AT(m, i, j) = sigmoid(act);
            }
        }
    }

    float sigmoid_dir(float z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }
    void mat_sigmoid_dir(Mat m) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) = sigmoid_dir(MAT_AT(m, i, j));
            }
        }
    }

    float Activation_ReLU(float z) {
        return MAX(0, z);
    }
    void mat_Activation_ReLU(Mat m) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) = Activation_ReLU(MAT_AT(m, i, j));
            }
        }
    }

    float Activation_ReLU_dir(float z) {
        return (z > 0) ? 1.0f : 0.0f;
    }

    void mat_Activation_ReLU_dir(Mat m) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) = Activation_ReLU_dir(MAT_AT(m, i, j));
            }
        }
    }

    void mat_Activation_Softmax(Mat m) {
        if (m.rows == 1) {
            MAT_AT(m, 0, 0) = sigmoid(MAT_AT(m, 0, 0));
            return;
        }

        mat_add_const(m, -(mat_max(m)));
        mat_exp(m);
        float sum = mat_sumof(m);
        mat_mul_const(m, 1.0f / sum);
    }

    void mat_normalized_tanh(Mat m) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) = (tanhf(MAT_AT(m, i, j)) + 1) / 2.0f;
            }
        }
    }

    void mat_normalized_tanh_dir(Mat m) {
        mat_normalized_tanh(m);
        mat_pow(m, 2);
        mat_mul_const(m, -1);
        mat_add_const(m, 1);
    }
    void mat_outputlayer_activation(Mat m) {
        mat_sigmoid(m);
        //mat_normalized_tanh(m);
        //mat_Activation_Softmax(m);
    }
    void mat_outputlayer_activation_dir(Mat m) {
        mat_sigmoid_dir(m);
        //mat_normalized_tanh_dir(m);
    }
    float outputlayer_activation(float z) {
        return sigmoid(z);
        //return tanhf(z);
        //return Activation_Softmax(z);
    }
    float outputlayer_activation_dir(float z) {
        return sigmoid_dir(z);
        //return (1 - (tanhf(z) * tanhf(z)));
    }




    float hiddenlayer_activation(float z) {
        //return Activation_ReLU(z);
        return sigmoid(z);
    }
    float hiddenlayer_activation_dir(float z) {
        //return Activation_ReLU_dir(z);
        return sigmoid_dir(z);
    }

    void mat_hiddenlayer_activation(Mat m) {
        //mat_Activation_ReLU(m);
        mat_sigmoid(m);
    }
    void mat_hiddenlayer_activation_dir(Mat m) {
        //mat_Activation_ReLU_dir(m);
        mat_sigmoid_dir(m);
    }



    float mse(Mat pred, Mat output) {
        float sum = 0;
        for (size_t i = 0; i < pred.rows; i++) {
            for (size_t j = 0; j < pred.cols; j++) {
                float temp = MAT_AT(pred, i, j) - MAT_AT(output, i, j);
                temp *= temp;
                sum += temp;
            }
        }
        return sum;
    }

    float crossentropy(Mat pred, Mat output) {
        float sum = 0;
        for (size_t i = 0; i < pred.rows; i++) {
            for (size_t j = 0; j < pred.cols; j++) {
                float a = MAT_AT(pred, i, j);
                if (a > 1 - 1e-7) a = 1.0f - 1e-7f;
                if (a < 1e-7) a = 1e-7f;
                float y = MAT_AT(output, i, j);
                sum += (y * logf(a) + (1 - y) * logf(1 - a));
            }
        }
        return -sum;
    }

    float mat_cost(Mat pred, Mat output) {
        assert(pred.rows == output.rows);
        assert(pred.cols == output.cols);
        return mse(pred, output);
        //return crossentropy(pred, output);
    }

    void feed_forward(NN nn) {

        Mat temp = NN_INPUT(nn);
        for (size_t i = 1; i < nn.count; i++) {
            mat_dot(nn.layers[i].z, temp, nn.layers[i].w);
            mat_addEW(nn.layers[i].z, nn.layers[i].z, nn.layers[i].b);

            mat_copy(nn.layers[i].a, nn.layers[i].z);
            if (i != nn.count - 1) {
                mat_hiddenlayer_activation(nn.layers[i].a);
            }
            else {
                mat_outputlayer_activation(nn.layers[i].a);
            }
            temp = nn.layers[i].a;
        }
    }

    float nn_cost(NN nn, Mat tinput, Mat toutput) {
        float sum = 0;
        for (size_t i = 0; i < tinput.rows; i++) {
            mat_copy(NN_INPUT(nn), mat_row(tinput, i));
            feed_forward(nn);
            //sum += mat_cost(NN_OUTPUT(nn), mat_row(toutput, i));
            for (size_t j = 0; j < toutput.cols; j++) {
                float act = MAT_AT(NN_OUTPUT(nn), 0, j);
                float d = act - MAT_AT(mat_row(toutput, i), 0, j);
                sum += d * d;
            }
        }
        sum = sum / (float)tinput.rows;

        return sum;
    }

    float cost_dir(float pred, float ttrue) {
        return 2 * (pred - ttrue);
    }





    void nn_rand(NN nn) {
        for (size_t i = 1; i < nn.count; i++) {
            mat_rand(nn.layers[i].w, 0, 1);
            mat_rand(nn.layers[i].b, 0, 1);
            mat_mul_const(nn.layers[i].w, (1.0f / sqrtf((float)nn.layers[i].w.rows)));
        }
    }

    Mat* mats_alloc(Arena* arena, NN nn, MatType mt) {
        Mat* mats = (Mat*)arena_alloc(arena, sizeof(Mat) * (nn.count - 1));
        assert(mats != NULL);
        if (mt == weights) {
            for (size_t i = 0; i < nn.count - 1; i++) {
                mats[i] = mat_alloc(arena, nn.layers[i + 1].w.rows, nn.layers[i + 1].w.cols);
            }
        }
        else if (mt == biases) {
            for (size_t i = 0; i < nn.count - 1; i++) {
                mats[i] = mat_alloc(arena, nn.layers[i + 1].b.rows, nn.layers[i + 1].b.cols);
            }
        }
        return mats;
    }

    Mat* gasalloc(Arena* arena, NN nn) {
        Mat* mats = (Mat*)arena_alloc(arena, sizeof(Mat) * (nn.count));
        assert(mats != NULL);
        for (size_t i = 0; i < nn.count; i++) {
            mats[i] = mat_alloc(arena, 1, nn.layers[i].a.cols);
        }
        return mats;
    }


    void reset_nablas(NN nn) {
        for (size_t i = 0; i < nn.count - 1; i++) {
            mat_fill(nabla_b[i], 0);
            mat_fill(nabla_w[i], 0);
        }
    }

    void finitediff(NN nn, Mat input, Mat output) {
        // approximate the derivative of the cost WRT each weight and bias
        float currcost = nn_cost(nn, input, output);

        // here we do the derivative of the cost WRT each weight
        for (size_t i = 1; i < nn.count; i++) {
            for (size_t j = 0; j < nn.layers[i].w.rows; j++) {
                for (size_t k = 0; k < nn.layers[i].w.cols; k++) {
                    float currw = MAT_AT(nn.layers[i].w, j, k);
                    MAT_AT(nn.layers[i].w, j, k) += eps;
                    float newcost = nn_cost(nn, input, output);
                    MAT_AT(nn.layers[i].w, j, k) = currw;
                    float temp = (newcost - currcost) / eps;
                    MAT_AT(nabla_w[i - 1], j, k) += temp;
                }
            }
        }
        // here we do the derivative of the cost WRT each bias
        for (size_t i = 1; i < nn.count; i++) {
            for (size_t j = 0; j < nn.layers[i].b.rows; j++) {
                for (size_t k = 0; k < nn.layers[i].b.cols; k++) {
                    float currb = MAT_AT(nn.layers[i].b, j, k);
                    MAT_AT(nn.layers[i].b, j, k) += eps;
                    float newcost = nn_cost(nn, input, output);
                    MAT_AT(nn.layers[i].b, j, k) = currb;
                    float temp = (newcost - currcost) / eps;
                    MAT_AT(nabla_b[i - 1], j, k) += temp;
                }
            }
        }
    }

    void backprop(Arena* arena, NN nn, Mat input, Mat output) {
        (void)arena;
        mat_copy(NN_INPUT(nn), input);
        feed_forward(nn);
        float ss = 2;
        for (size_t j = 0; j < nn.layers[nn.count - 1].a.cols; ++j) {
            float a = MAT_AT(nn.layers[nn.count - 1].a, 0, j);
            float da = ss * (a - MAT_AT(output, 0, j));
            float qa = outputlayer_activation_dir(MAT_AT(nn.layers[nn.count - 1].z, 0, j));
            MAT_AT(nabla_b[nn.count - 2], 0, j) += ss * da * qa;
            for (size_t k = 0; k < nn.layers[nn.count - 2].a.cols; ++k) {
                // j - weight matrix col
                // k - weight matrix row
                float pa = MAT_AT(nn.layers[nn.count - 2].a, 0, k);
                MAT_AT(nabla_w[nn.count - 2], k, j) += ss * da * qa * pa;
            }
        }

        for (size_t l = nn.count - 2; l > 0; --l) {
            for (size_t j = 0; j < nn.layers[l].a.cols; ++j) {
                float temp = 0;
                for (size_t i = 0; i < nabla_b[l].cols; i++) {
                    temp += MAT_AT(nabla_b[l], 0, i) * MAT_AT(nn.layers[l + 1].w, j, i);
                }
                float da = temp;
                float qa = hiddenlayer_activation_dir(MAT_AT(nn.layers[l].z, 0, j));
                MAT_AT(nabla_b[l - 1], 0, j) += ss * da * qa;
                for (size_t k = 0; k < nn.layers[l - 1].a.cols; ++k) {
                    // j - weight matrix col
                    // k - weight matrix row
                    float pa = MAT_AT(nn.layers[l - 1].a, 0, k);
                    MAT_AT(nabla_w[l - 1], k, j) += ss * da * qa * pa;
                }
            }
        }
    }
    void update_mini_batch(Arena* arena, NN nn, Mat mini_batchin, Mat mini_batchout, float LearRate, float RegParam, size_t n) {

        for (size_t i = 0; i < mini_batchin.rows; i++) {
            Mat input = mat_row(mini_batchin, i);
            Mat output = mat_row(mini_batchout, i);
#if BP
            backprop(arena, nn, input, output);
#else 
            finitediff(nn, input, output);
#endif
        }
        for (size_t i = 0; i < nn.count - 1; i++) {
            mat_mul_const(nabla_w[i], LearRate / (float)mini_batchin.rows);
            mat_mul_const(nabla_b[i], LearRate / (float)mini_batchin.rows);
            mat_mul_const(nn.layers[i + 1].w, (1 - LearRate * (RegParam / n)));
            mat_subEW(nn.layers[i + 1].w, nn.layers[i + 1].w, nabla_w[i]);
            mat_subEW(nn.layers[i + 1].b, nn.layers[i + 1].b, nabla_b[i]);
        }
        reset_nablas(nn);
    }


    void learn(Arena* arena, NN nn, Mat traininput, Mat trainoutput, size_t epochs, size_t mini_batch_size, float LearRate, float RegParam) {

        size_t n = traininput.rows;
        size_t batches = n / mini_batch_size;
        nabla_b = mats_alloc(arena, nn, (MatType)biases);
        nabla_w = mats_alloc(arena, nn, (MatType)weights);
        for (size_t i = 0; i < epochs; i++) {
            for (size_t j = 0; j < batches; j++) {
                Mat mini_batchin = mat_mat(traininput, j, j + (mini_batch_size - 1), 0, traininput.cols - 1);
                Mat mini_batchout = mat_mat(trainoutput, j, j + (mini_batch_size - 1), 0, trainoutput.cols - 1);
                update_mini_batch(arena, nn, mini_batchin, mini_batchout, LearRate, RegParam, n);
            }
            printf("cost : %f\n", nn_cost(nn, traininput, trainoutput));
        }
    }

}


