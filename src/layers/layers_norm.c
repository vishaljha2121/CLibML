//
// Created by Vishal Jha on 16/01/26.
//
#include "../../include/layers.h"
#include "layers_internal.h"

#include <math.h>

// Returns standard deviation
f32 _norm_backend(tensor* t, f32 epsilon);

void _layer_norm_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);

    out->shape = prev_shape;

    out->norm_backend.epsilon = desc->norm.epsilon;
}
void _layer_norm_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    f32 std_dev = _norm_backend(in_out, l->norm_backend.epsilon);

    // TODO: store this without the tensor cache
    // Very dumb with GPU backend
    if (cache != NULL && l->training_mode) {
        tensor* stdv = tensor_create(cache->arena, (tensor_shape){ 1, 1, 1 });
        tensor_set_data(stdv, &std_dev);
        
        layers_cache_push(cache, stdv);
    }
}
void _layer_norm_backprop(layer* l, tensor* delta, layers_cache* cache) {
    UNUSED(l);

    tensor* stdv = layers_cache_pop(cache);
    f32 std_dev = 1.0f;
    tensor_get_data(&std_dev, stdv);

    tensor_scale_ip(delta, delta, 1.0f / std_dev);
}

#if TENSOR_BACKEND == TENSOR_BACKEND_CPU

f32 _norm_backend(tensor* t, f32 epsilon) {
    u64 size = (u64)t->shape.width * t->shape.height * t->shape.depth;

    f32* data = (f32*)t->data;

    float mean = 0.0f;
    for (u64 i = 0; i < size; i++) {
        mean += data[i];
    }
    mean /= size;

    float std_dev = 0.0f;
    for (u64 i = 0; i < size; i++) {
        std_dev += (data[i] - mean) * (data[i] - mean);
    }
    std_dev = sqrtf((std_dev / size) + epsilon);

    for (u64 i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std_dev;
    }

    return std_dev;
}

#endif // TENSOR_BACKEND == TENSOR_BACKEND_CPU