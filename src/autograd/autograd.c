//
// Created by Vishal Jha on 13/01/26.
//

#include "autograd.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

b32 mat_relu(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = MAX(0, in->data[i]);
    }

    return true;
}

b32 mat_softmax(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i]);
        sum += out->data[i];
    }

    mat_scale(out, 1.0f / sum);

    return true;
}

b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    if (p->rows != q->rows || p->cols != q->cols) { return false; }
    if (out->rows != p->rows || out->cols != p->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = p->data[i] == 0.0f ?
            0.0f : p->data[i] * -logf(q->data[i]);
    }

    return true;
}

b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }
    if (out->rows != grad->rows || out->cols != grad->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    }

    return true;
}

b32 mat_softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
) {
    if (softmax_out->rows != 1 && softmax_out->cols != 1) {
        return false;
    }

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    u32 size = MAX(softmax_out->rows, softmax_out->cols);
    matrix* jacobian = mat_create(scratch.arena, size, size);

    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j++) {
            jacobian->data[j + i * size] =
                softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
        }
    }

    mat_mul(out, jacobian, grad, 0, 0, 0);

    arena_scratch_release(scratch);

    return true;
}

b32 mat_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
) {
    if (p->rows != q->rows || p->cols != q->cols) { return false; }

    u64 size = (u64)p->rows * p->cols;

    if (p_grad != NULL) {
        if (p_grad->rows != p->rows || p_grad->cols != p->cols) {
            return false;
        }

        for (u64 i = 0; i < size; i++) {
            p_grad->data[i] += -logf(q->data[i]) * grad->data[i];
        }
    }

    if (q_grad != NULL) {
        if (q_grad->rows != q->rows || q_grad->cols != q->cols) {
            return false;
        }

        for (u64 i = 0; i < size; i++) {
            q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
        }
    }

    return true;
}