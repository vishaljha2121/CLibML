//
// Created by Vishal Jha on 13/01/26.
//

#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include "../base.h"
#include "../tensor/tensor.h"

b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
b32 mat_softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
);
b32 mat_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
);
#endif //AUTOGRAD_H
