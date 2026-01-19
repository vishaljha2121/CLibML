//
// Created by Vishal Jha on 16/01/26.
//
#include "../../include/layers.h"
#include "layers_internal.h"
#include "../random_generators/prng.h"

#include <stdio.h>
#include <stdlib.h>

void _make_dropout_tensor(tensor* out, f32 keep_rate);

void _layer_dropout_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);

    layer_dropout_backend* dropout = &out->dropout_backend;

    dropout->keep_rate = desc->dropout.keep_rate;

    out->shape = prev_shape;
}
void _layer_dropout_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    if (l->training_mode && cache != NULL) {
        f32 keep_rate = l->dropout_backend.keep_rate;

        // Creating dropout tensor
        tensor* dropout_tensor = tensor_create(cache->arena, in_out->shape);
        _make_dropout_tensor(dropout_tensor, keep_rate);

        // Applying tensor to input
        tensor_component_mul_ip(in_out, in_out, dropout_tensor);
        tensor_scale_ip(in_out, in_out, 1.0f / keep_rate);

        // Saving dropout_tensor in cache
        layers_cache_push(cache, dropout_tensor);
    }
}
void _layer_dropout_backprop(layer* l, tensor* delta, layers_cache* cache) {
    f32 keep_rate = l->dropout_backend.keep_rate;

    tensor* dropout_tensor = layers_cache_pop(cache);

    tensor_component_mul_ip(delta, delta, dropout_tensor);
    tensor_scale_ip(delta, delta, 1.0f / keep_rate);
}

#if TENSOR_BACKEND == TENSOR_BACKEND_CPU

void _make_dropout_tensor(tensor* out, f32 keep_rate) {
    tensor_shape s = out->shape;
    u64 size = (u64)s.width * s.height * s.depth;

    f32* data = (f32*)out->data;

    for (u64 i = 0; i < size; i++) {
        data[i] = prng_rand_f32() > keep_rate ? 0.0f : 1.0f;
    }
}

#endif // TENSOR_BACKEND == TENSOR_BACKEND_CPU