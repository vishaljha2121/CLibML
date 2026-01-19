//
// Created by Vishal Jha on 16/01/26.
//
#include "../../include/cost.h"
#include "../../include/err.h"

#include <stdio.h>
#include <math.h>

typedef f32 (_cost_func)(const tensor*, const tensor*);
typedef void (_cost_grad)(tensor*, const tensor*);

typedef struct {
    _cost_func* func;
    _cost_grad* grad;
} _cost;

static f32 null_func(const tensor* in, const tensor* desired_out);
static void null_grad(tensor* in_out, const tensor* desired_out);
static f32 mean_squared_func(const tensor* in, const tensor* desired_out);
static void mean_squared_grad(tensor* in_out, const tensor* desired_out);
static f32 cce_func(const tensor* in, const tensor* desired_out);
static void cce_grad(tensor* in_out, const tensor* desired_out);

static _cost _costs[COST_COUNT] = {
    [COST_NULL] = { null_func, null_grad },
    [COST_MEAN_SQUARED_ERROR] = { mean_squared_func, mean_squared_grad },
    [COST_CATEGORICAL_CROSS_ENTROPY] = { cce_func, cce_grad },
};

f32 cost_func(cost_type type, const tensor* in, const tensor* desired_out) {
    // Checks
    if (type >= COST_COUNT) {
        ERR(ERR_INVALID_INPUT, "Invalid cost function type");
        return 0.0f;
    }
    if (!tensor_shape_eq(in->shape, desired_out->shape)) {
        ERR(ERR_INVALID_INPUT, "Input and desired output must align in cost function");
        return 0.0f;
    }
    if (in == NULL || desired_out == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot compute cost: in and/or desired_out is NULL");
        return 0.0f;
    }

    // Actual cost function
    return _costs[type].func(in, desired_out);
}
void cost_grad(cost_type type, tensor* in_out, const tensor* desired_out) {
    // Checks
    if (type >= COST_COUNT) {
        ERR(ERR_INVALID_INPUT, "Invalid cost function type");
        return;
    }
    if (!tensor_shape_eq(in_out->shape, desired_out->shape)) {
        ERR(ERR_INVALID_INPUT, "Input and desired output must align in cost function");
        return;
    }
    if (in_out == NULL || desired_out == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot compute cost: in and/or desired_out is NULL");
        return;
    }

    // Actual gradient
    _costs[type].grad(in_out, desired_out);
}

static f32 null_func(const tensor* in, const tensor* desired_out) {
    UNUSED(in);
    UNUSED(desired_out);

    return 0.0f;
}
static void null_grad(tensor* in_out, const tensor* desired_out) {
    UNUSED(in_out);
    UNUSED(desired_out);
}

#if TENSOR_BACKEND == TENSOR_BACKEND_CPU

static f32 mean_squared_func(const tensor* in, const tensor* desired_out) {
    f32 sum = 0.0f;

    f32* in_data = (f32*)in->data;
    f32* d_out_data = (f32*)desired_out->data;

    u64 size = (u64)in->shape.width * in->shape.height * in->shape.depth;
    for (u64 i = 0; i < size; i++) {
        sum += 0.5f * (in_data[i] - d_out_data[i]) * (in_data[i] - d_out_data[i]);
    }

    return sum / (f32)size;
}
static void mean_squared_grad(tensor* in_out, const tensor* desired_out) {
    u64 size = (u64)in_out->shape.width * in_out->shape.height * in_out->shape.depth;

    f32* in_out_data = (f32*)in_out->data;
    f32* d_out_data = (f32*)desired_out->data;

    for (u64 i = 0; i < size; i++) {
        in_out_data[i] = (in_out_data[i] - d_out_data[i]);
    }
}

static f32 cce_func(const tensor* in, const tensor* desired_out) {
    f32 sum = 0.0f;

    f32* in_data = (f32*)in->data;
    f32* d_out_data = (f32*)desired_out->data;

    u64 size = (u64)in->shape.width * in->shape.height * in->shape.depth;
    for (u64 i = 0; i < size; i++) {
        sum += d_out_data[i] * logf(in_data[i]);
    }

    return -sum;
}
static void cce_grad(tensor* in_out, const tensor* desired_out) {
    u64 size = (u64)in_out->shape.width * in_out->shape.height * in_out->shape.depth;

    f32* in_out_data = (f32*)in_out->data;
    f32* d_out_data = (f32*)desired_out->data;

    for (u64 i = 0; i < size; i++) {
        in_out_data[i] = -d_out_data[i] / (in_out_data[i] + 1e-8f);
    }
}

#endif // TENSOR_BACKEND == TENSOR_BACKEND_CPU