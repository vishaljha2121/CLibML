//
// Created by Vishal Jha on 16/01/26.
//

#ifndef LAYERS_INTERNAL_H
#define LAYERS_INTERNAL_H

#include "../../include/layers.h"


// TODO: consistent underscoring for private stuff
typedef struct {
    layer_create_func* create;
    layer_feedforward_func* feedforward;
    layer_backprop_func* backprop;
    layer_apply_changes_func* apply_changes;
    layer_delete_func* delete;
    layer_save_func* save;
    layer_load_func* load;
} _layer_func_defs;

// These functions are implemented in specific layers_*.c files

void _layer_null_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_null_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_null_backprop(layer* l, tensor* delta, layers_cache* cache);
void _layer_null_apply_changes(layer* l, const optimizer* optim);
void _layer_null_delete(layer* l);
void _layer_null_save(mg_arena* arena, layer* l, tensor_list* list, u32 index);
void _layer_null_load(layer* l, const tensor_list* list, u32 index);

void _layer_input_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape); 
void _layer_input_feedforward(layer* l, tensor* in_out, layers_cache* cache);

void _layer_reshape_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_reshape_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_reshape_backprop(layer* l, tensor* delta, layers_cache* cache);

void _layer_dense_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_dense_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_dense_backprop(layer* l, tensor* delta, layers_cache* cache);
void _layer_dense_apply_changes(layer* l, const optimizer* optim);
void _layer_dense_delete(layer* l);
void _layer_dense_save(mg_arena* arena, layer* l, tensor_list* list, u32 index);
void _layer_dense_load(layer* l, const tensor_list* list, u32 index);

void _layer_activation_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_activation_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_activation_backprop(layer* l, tensor* delta, layers_cache* cache);

void _layer_dropout_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_dropout_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_dropout_backprop(layer* l, tensor* delta, layers_cache* cache);

void _layer_flatten_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_flatten_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_flatten_backprop(layer* l, tensor* delta, layers_cache* cache);

void _layer_pooling_2d_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_pooling_2d_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_pooling_2d_backprop(layer* l, tensor* delta, layers_cache* cache);

void _layer_conv_2d_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_conv_2d_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_conv_2d_backprop(layer* l, tensor* delta, layers_cache* cache);
void _layer_conv_2d_apply_changes(layer* l, const optimizer* optim);
void _layer_conv_2d_delete(layer* l);
void _layer_conv_2d_save(mg_arena* arena, layer* l, tensor_list* list, u32 index);
void _layer_conv_2d_load(layer* l, const tensor_list* list, u32 index);

void _layer_norm_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_norm_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_norm_backprop(layer* l, tensor* delta, layers_cache* cache);

#endif // LAYERS_INTERNAL_H