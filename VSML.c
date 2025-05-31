#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "assert.h"
#include "math.h"

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float* es;
} Mat;

typedef struct {
    Mat w;
    Mat b;
    Mat a;
} Layer;

typedef struct {
    Layer* layers;
    size_t count;
} NN;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]
#define MIN(a,b) (((a)<(b)) ? (a) : (b))
#define MAX(a,b) (((a)>(b)) ? (a) : (b))
#define ARRAY_LEN(arr) ((sizeof(arr)) / (sizeof((arr)[0])))

#define nob_da_append(da, item)                                                          \
    do {                                                                                 \
        if ((da)->count >= (da)->capacity) {                                             \
            (da)->capacity = (da)->capacity == 0 ? 8 : (da)->capacity*2;   \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                       \
        }                                                                                \
                                                                                         \
        (da)->items[(da)->count++] = (item);                                             \
    } while (0)

#define MAT_PRINT(m) mat_print(m, #m)
#define LAY_PRINT(l) lay_print(l, #l)
#define NN_PRINT(nn) nn_print(nn, #nn)
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

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = malloc(rows * cols * sizeof(*m.es));
    assert(m.es != NULL);
    return m;
}

Mat mat_dup(Mat m) {
    Mat copy;
    copy.rows = m.rows;
    copy.cols = m.cols;
    copy.stride = m.stride;
    copy.es = malloc(copy.rows * copy.cols * sizeof(*copy.es));
    assert(copy.es != NULL);
    for (size_t i = 0; i < copy.rows; i++) {
        for (size_t j = 0; j < copy.cols; j++) {
            MAT_AT(copy, i, j) = MAT_AT(m, i, j);
        }
    }
    return copy;
}

Mat mat_dup_col(Mat m, size_t c) {
    Mat temp = mat_alloc(m.rows, 1);
    for (size_t j = 0; j < temp.rows; j++) {
        MAT_AT(temp, j, 0) = MAT_AT(m, j, c);
    }
    return temp;
}

void mat_copy(Mat dst, Mat src) {
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

Mat mat_row(Mat m, size_t r) {
    return (Mat) {
        .rows = 1,
            .cols = m.cols,
            .stride = m.stride,
            .es = &MAT_AT(m, r, 0)
    };
}

Mat mat_mat(Mat m, size_t sr, size_t er, size_t sc, size_t ec) {
    Mat temp = mat_alloc(er - sr + 1, ec - sc + 1);
    for (size_t i = 0; i < temp.rows; i++) {
        for (size_t j = 0; j < temp.cols; j++) {
            MAT_AT(temp, i, j) = MAT_AT(m, i, j);
        }
    }
    return temp;
}

Mat mat_dup_row(Mat m, size_t r) {
    return mat_dup(mat_row(m, 0));
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
            MAT_AT(c, i, j) = MAT_AT(a, i, j) + MAT_AT(b, i, j);
        }
    }
}

void mat_subEW(Mat c, Mat a, Mat b) {
    assert(c.rows == a.rows && a.rows == b.rows);
    assert(c.cols == a.cols && a.cols == b.cols);
    for (size_t i = 0; i < c.rows; i++) {
        for (size_t j = 0; j < c.cols; j++) {
            MAT_AT(c, i, j) = MAT_AT(a, i, j) - MAT_AT(b, i, j);
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
            MAT_AT(m, i, j) = pow(MAT_AT(m, i, j), x);
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
    float temp = 0;
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            if (MAT_AT(m, i, j) > temp)
                temp = MAT_AT(m, i, j);
        }
    }
    return temp;
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
    for (size_t i = 0; i < nn.count; i++) {
        LAY_PRINT(nn.layers[i]);
    }
    printf("]\n");
}


float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}
void mat_sigmoid(Mat m) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = sigmoid(MAT_AT(m, i, j));
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
    return (z > 0) ? 1 : 0;
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
    Mat exppos = mat_dup(m);
    Mat expneg = mat_dup(m);

    mat_mul_const(expneg, -1);
    mat_exp(exppos);
    mat_exp(expneg);
    mat_subEW(m, exppos, expneg);
    mat_addEW(exppos, exppos, expneg);
    mat_divEW(m, m, exppos);
    mat_add_const(m, 1);
    mat_mul_const(m, 0.5f);
}
void mat_normalized_tanh_dir(Mat m) {
    mat_normalized_tanh(m);
    mat_pow(m, 2);
    mat_mul_const(m, -1);
    mat_add_const(m, 1);
}


void outputlayer_activation(Mat m) {
    //Activation_Softmax(m);
    mat_normalized_tanh(m);
}
void hiddenlayer_activation(Mat m) {
    mat_Activation_ReLU(m);
    //Activation_LeakyReLU(m);
    //normalized_tanh(m);
}
void hiddenlayer_activation_dir(Mat m) {
    mat_Activation_ReLU_dir(m);
    //Activation_LeakyReLU_dir(m);
    //normalized_tanh_dir(m);
}


NN nn_alloc(size_t* nn_struct, size_t count) {
    NN nn;
    nn.count = count - 1;
    nn.layers = malloc(count * sizeof(*nn.layers));
    for (size_t i = 0; i < nn.count; i++) {
        nn.layers[i].w = mat_alloc(nn_struct[i], nn_struct[i + 1]);
        nn.layers[i].b = mat_alloc(1, nn_struct[i + 1]);
        nn.layers[i].a = mat_alloc(1, nn_struct[i + 1]);
    }
    return nn;
}


float train[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};


int main(void) {
    srand(time(0));
    //size_t stride = 3;
    //size_t r = ARRAY_LEN(train)/stride;

    size_t nn_struct[] = { 2, 2, 1 };
    size_t count = ARRAY_LEN(nn_struct);


    NN nn = nn_alloc(nn_struct, count);
    NN_PRINT(nn);
    //MAT_AT(nn.layers[0].w, 0, 0) = 123;
    NN_PRINT(nn);
    getchar();

    return 0;
}
