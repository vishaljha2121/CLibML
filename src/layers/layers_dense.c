//
// Created by Vishal Jha on 16/01/26.
//
#include "../../include/layers.h"
#include "layers_internal.h"

#include <stdlib.h>
#include <math.h>

// TODO: remove
#include <stdio.h>

void _layer_dense_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    u32 in_size = prev_shape.width;
    u32 out_size = desc->dense.size;

    tensor_shape bias_shape = { out_size, 1, 1 };
    tensor_shape weight_shape = { out_size, in_size, 1 };

    out->shape = bias_shape;

    layer_dense_backend* dense = &out->dense_backend;

    dense->bias = tensor_create(arena, bias_shape);
    dense->weight = tensor_create(arena, weight_shape);

    if (out->training_mode) {
        param_change_create(arena, &dense->bias_change, bias_shape);
        param_change_create(arena, &dense->weight_change, weight_shape);
    }

    param_init(dense->bias, desc->dense.bias_init, in_size, out_size);
    param_init(dense->weight, desc->dense.weight_init, in_size, out_size);
}
void _layer_dense_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    layer_dense_backend* dense = &l->dense_backend;

    if (cache != NULL && l->training_mode) {
        tensor* input = tensor_copy(cache->arena, in_out, false);
        layers_cache_push(cache, input);
    }

    tensor_dot_ip(in_out, false, false, in_out, dense->weight);
    tensor_add_ip(in_out, in_out, dense->bias);
}
void _layer_dense_backprop(layer* l, tensor* delta, layers_cache* cache) {
    layer_dense_backend* dense = &l->dense_backend;

    // Bias change is just delta
    param_change_add(&dense->bias_change, delta);

    // Weight change is previous input dotted with delta
    // weight_change += dot(prev_input, delta)
    mga_temp scratch = mga_scratch_get(&cache->arena, 1);

    tensor* prev_input = layers_cache_pop(cache);

    tensor* cur_weight_change = tensor_dot(scratch.arena, true, false, prev_input, delta);
    param_change_add(&dense->weight_change, cur_weight_change);

    mga_scratch_release(scratch);

    // Delta is updated by weight
    // delta = dot(delta, transpose(weight))
    tensor_dot_ip(delta, false, true, delta, dense->weight);
}
void _layer_dense_apply_changes(layer* l, const optimizer* optim) {
    layer_dense_backend* dense = &l->dense_backend;

    param_change_apply(optim, dense->weight, &dense->weight_change);
    param_change_apply(optim, dense->bias, &dense->bias_change);
}
void _layer_dense_delete(layer* l) {
    layer_dense_backend* dense = &l->dense_backend;

    if (l->training_mode) {
        param_change_delete(&dense->weight_change);
        param_change_delete(&dense->bias_change);
    }
}

typedef struct {
    string8 weight_name;
    string8 bias_name;
} _param_names;

_param_names _get_param_names(mg_arena* arena, u32 index) {
    _param_names out = { 0 };

    out.weight_name = str8_pushf(arena, "dense_weight_%u", index);
    out.bias_name = str8_pushf(arena, "dense_bias_%u", index);

    return out;
}

void _layer_dense_save(mg_arena* arena, layer* l, tensor_list* list, u32 index) {
    layer_dense_backend* dense = &l->dense_backend;

    _param_names names = _get_param_names(arena, index);

    tensor_list_push(arena, list, dense->weight, names.weight_name);
    tensor_list_push(arena, list, dense->bias, names.bias_name);
}
void _layer_dense_load(layer* l, const tensor_list* list, u32 index) {
    layer_dense_backend* dense = &l->dense_backend;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    _param_names names = _get_param_names(scratch.arena, index);

    tensor* loaded_weight = tensor_list_get(list, names.weight_name);
    tensor* loaded_bias = tensor_list_get(list, names.bias_name);

    tensor_copy_ip(dense->weight, loaded_weight);
    tensor_copy_ip(dense->bias, loaded_bias);

    mga_scratch_release(scratch);
}