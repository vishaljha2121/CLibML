//
// Created by Vishal Jha on 16/01/26.
//

#include "../../include/optimizers.h"

#include <stdio.h>
#include <math.h>

void param_change_create(mg_arena* arena, param_change* out, tensor_shape shape) {
    out->_mutex = mutex_create(arena);

    out->_change = tensor_create(arena, shape);
    out->_V = tensor_create(arena, shape);
    out->_S = tensor_create(arena, shape);
}
void param_change_add(param_change* param_change, tensor* addend) {
    mutex_lock(param_change->_mutex);

    tensor_add_ip(param_change->_change, param_change->_change, addend);

    mutex_unlock(param_change->_mutex);
}
void param_change_delete(param_change* param_change) {
    mutex_destroy(param_change->_mutex);
}

typedef void(_param_apply_func)(const optimizer*, tensor*, param_change*);

void _null_param_apply(const optimizer* optim, tensor* param, param_change* param_change);
void _sgd_param_apply(const optimizer* optim, tensor* param, param_change* param_change);
void _rms_prop_param_apply(const optimizer* optim, tensor* param, param_change* param_change);
void _adam_param_apply(const optimizer* optim, tensor* param, param_change* param_change);

static _param_apply_func* _apply_funcs[OPTIMIZER_COUNT] = {
    [OPTIMIZER_NULL] = _null_param_apply,
    [OPTIMIZER_SGD] = _sgd_param_apply,
    [OPTIMIZER_RMS_PROP] = _rms_prop_param_apply,
    [OPTIMIZER_ADAM] = _adam_param_apply,
};

void param_change_apply(const optimizer* optim, tensor* param, param_change* param_change) {
    if (optim->type >= OPTIMIZER_COUNT) {
        // TODO: make ERR
        fprintf(stderr, "Cannot update param: Invalid optimizer type\n");
        return;
    }

    mutex_lock(param_change->_mutex);

    _apply_funcs[optim->type](optim, param, param_change);
    tensor_fill(param_change->_change, 0.0f);

    mutex_unlock(param_change->_mutex);
}

void _null_param_apply(const optimizer* optim, tensor* param, param_change* param_change) {
    UNUSED(optim);
    UNUSED(param);
    UNUSED(param_change);
}
void _sgd_param_apply(const optimizer* optim, tensor* param, param_change* param_change) {
    f32 beta = optim->sgd.momentum;

    // Averaging change over batch 
    tensor_scale_ip(param_change->_change, param_change->_change, 1.0f / (f32)optim->_batch_size);

    // V_t = beta * V_t-1 + (1 - beta) * d
    tensor_scale_ip(param_change->_V, param_change->_V, beta);
    tensor_scale_ip(param_change->_change, param_change->_change, 1.0f - beta);
    tensor_add_ip(param_change->_V, param_change->_V, param_change->_change);

    // param = param - (learning_rate * V)
    tensor_scale_ip(param_change->_change, param_change->_V, optim->learning_rate);
    tensor_sub_ip(param, param, param_change->_change);
}
void _rms_prop_param_apply(const optimizer* optim, tensor* param, param_change* param_change) {
    f32 beta = optim->rms_prop.beta;
    f32 epsilon = optim->rms_prop.epsilon;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Averaging change over batch 
    tensor_scale_ip(param_change->_change, param_change->_change, 1.0f / (f32)optim->_batch_size);
    tensor* real_change = tensor_copy(scratch.arena, param_change->_change, false);

    if (tensor_is_zero(param_change->_S)) {
        // S_0 = d^2
        tensor_component_mul_ip(param_change->_change, param_change->_change, param_change->_change);
        tensor_copy_ip(param_change->_S, param_change->_change);

    } else {
        // S_t = beta * S_t-1 + (1 - beta) * d^2
        tensor_scale_ip(param_change->_S, param_change->_S, beta);
        tensor_component_mul_ip(param_change->_change, param_change->_change, param_change->_change);
        tensor_scale_ip(param_change->_change, param_change->_change, 1.0f - beta);
        tensor_add_ip(param_change->_S, param_change->_S, param_change->_change);
    }

    // param = param - (learning_rate / sqrt(S + epsilon)) * dW
    tensor* sqrt_S = tensor_copy(scratch.arena, param_change->_S, false);
    tensor_add_all_ip(sqrt_S, sqrt_S, epsilon);
    tensor_sqrt_ip(sqrt_S, sqrt_S);

    tensor_component_div_ip(real_change, real_change, sqrt_S);
    tensor_scale_ip(real_change, real_change, optim->learning_rate);

    tensor_sub_ip(param, param, real_change);

    mga_scratch_release(scratch);
}
void _adam_param_apply(const optimizer* optim, tensor* param, param_change* param_change) {
    optimizer_adam adam = optim->adam;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Averaging change over batch 
    tensor_scale_ip(param_change->_change, param_change->_change, 1.0f / (f32)optim->_batch_size);
    tensor* real_change = tensor_copy(scratch.arena, param_change->_change, false);

    // V_t = beta * V_t-1 + (1 - beta) * d
    tensor_scale_ip(param_change->_V, param_change->_V, adam.beta1);
    tensor_scale_ip(param_change->_change, param_change->_change, 1.0f - adam.beta1);
    tensor_add_ip(param_change->_V, param_change->_V, param_change->_change);

    // Putting original change back in change->_change
    tensor_copy_ip(param_change->_change, real_change);

    // S_t = beta * S_t-1 + (1 - beta) * d^2
    tensor_scale_ip(param_change->_S, param_change->_S, adam.beta2);
    tensor_component_mul_ip(param_change->_change, param_change->_change, param_change->_change);
    tensor_scale_ip(param_change->_change, param_change->_change, 1.0f - adam.beta2);
    tensor_add_ip(param_change->_S, param_change->_S, param_change->_change);

    // param = param - (learning_rate / sqrt(S + epsilon)) * V
    tensor* sqrt_S = tensor_copy(scratch.arena, param_change->_S, false);
    tensor_add_all_ip(sqrt_S, sqrt_S, adam.epsilon);
    tensor_sqrt_ip(sqrt_S, sqrt_S);

    tensor_copy_ip(real_change, param_change->_V);

    tensor_component_div_ip(real_change, real_change, sqrt_S);
    tensor_scale_ip(real_change, real_change, optim->learning_rate);

    tensor_sub_ip(param, param, real_change);

    mga_scratch_release(scratch);
}