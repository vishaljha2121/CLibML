//
// Created by Vishal Jha on 16/01/26.
//
#include "../../include/layers.h"
#include "layers_internal.h"

void _layer_conv_2d_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    const layer_conv_2d_desc* cdesc = &desc->conv_2d;
    layer_conv_2d_backend* conv = &out->conv_2d_backend;

    conv->kernel_size = cdesc->kernel_size;
    conv->stride = cdesc->stride;
    conv->input_shape = prev_shape;

    if (cdesc->padding) {
        // Padding so that out_shape == in_shape when stride is 1
        conv->padding = (conv->kernel_size - 1) / 2;
    }

    tensor_shape padded_shape = (tensor_shape){
        prev_shape.width + conv->padding * 2,
        prev_shape.height + conv->padding * 2,
        prev_shape.depth
    };
    out->shape = tensor_conv_shape(padded_shape, (tensor_shape){ conv->kernel_size, conv->kernel_size, 1 }, conv->stride, conv->stride);
    out->shape.depth = cdesc->num_filters;

    // Have to collapse one dimension because tensors are only 3d
    tensor_shape kernels_shape = {
        .width = conv->kernel_size * conv->kernel_size * prev_shape.depth,
        .height = cdesc->num_filters,
        .depth = 1
    };

    conv->kernels = tensor_create(arena, kernels_shape);
    conv->biases = tensor_create(arena, out->shape);

    u64 in_size = (u64)prev_shape.width * prev_shape.height * prev_shape.depth;
    u64 out_size = (u64)out->shape.width * out->shape.height * out->shape.depth;
    param_init(conv->kernels, cdesc->kernels_init, in_size, out_size);
    param_init(conv->biases, cdesc->biases_init, in_size, out_size);

    if (desc->training_mode) { 
        param_change_create(arena, &conv->kernels_change, kernels_shape);
        param_change_create(arena, &conv->biases_change, out->shape);
    }
}

void _layer_conv_2d_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    mga_temp scratch = { 0 };
    mg_arena* col_arena = NULL;
    if (cache != NULL) {
        col_arena = cache->arena;
    } else {
        scratch = mga_scratch_get(NULL, 0);
        col_arena = scratch.arena;
    }

    // Article explaining how to turn conv into mat mul
    // https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/

    // Renaming for clarity
    tensor* output = in_out;

    tensor* input_cols = tensor_im2col(col_arena, in_out, conv->kernel_size, conv->stride, conv->padding);
    if (cache != NULL) {
        layers_cache_push(cache, input_cols);
    }

    tensor_dot_ip(output, false, false, conv->kernels, input_cols);

    output->shape = l->shape;

    tensor_add_ip(output, output, conv->biases);

    if (scratch.arena != NULL) {
        mga_scratch_release(scratch);
    }
}

void _layer_conv_2d_backprop(layer* l, tensor* delta, layers_cache* cache) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    // Biases change is just delta
    param_change_add(&conv->biases_change, delta);

    tensor* input_cols = layers_cache_pop(cache);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Updating kernels
    // kernel_change = delta * prev_input
    tensor delta_view = (tensor){
        .shape = (tensor_shape){
            delta->shape.width * delta->shape.height,
            delta->shape.depth, 1
        },
        .data = delta->data
    };
    
    tensor* kernels_change = tensor_create(
        scratch.arena,
        (tensor_shape){
            conv->kernels->shape.width * conv->kernels->shape.height,
            conv->kernels->shape.depth, 1
        }
    );

    
    tensor_dot_ip(kernels_change, false, true, &delta_view, input_cols);
    param_change_add(&conv->kernels_change, kernels_change);

    // Resetting scratch after kernels change
    mga_temp_end(scratch);

    // Updating delta
    // delta *= kernels
    // Math is done in columns, then converted back to an image

    tensor* delta_cols = tensor_dot(scratch.arena, true, false, conv->kernels, &delta_view);
    tensor_col2im_ip(delta, delta_cols, conv->input_shape, conv->kernel_size, conv->stride, conv->padding);

    mga_scratch_release(scratch);
}
void _layer_conv_2d_apply_changes(layer* l, const optimizer* optim) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    param_change_apply(optim, conv->kernels, &conv->kernels_change);
    param_change_apply(optim, conv->biases, &conv->biases_change);
}
void _layer_conv_2d_delete(layer* l) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    if (l->training_mode) {
        param_change_delete(&conv->kernels_change);
        param_change_delete(&conv->biases_change);
    }
}
void _layer_conv_2d_save(mg_arena* arena, layer* l, tensor_list* list, u32 index) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    string8 kernels_name = str8_pushf(arena, "conv_2d_kernels_%u", index);
    string8 biases_name = str8_pushf(arena, "conv_2d_biases_%u", index);

    tensor_list_push(arena, list, conv->kernels, kernels_name);
    tensor_list_push(arena, list, conv->biases, biases_name);
}
void _layer_conv_2d_load(layer* l, const tensor_list* list, u32 index) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    string8 kernels_name = str8_pushf(scratch.arena, "conv_2d_kernels_%u", index);
    string8 biases_name = str8_pushf(scratch.arena, "conv_2d_biases_%u", index);

    tensor* loaded_kernels = tensor_list_get(list, kernels_name);
    tensor* loaded_biases = tensor_list_get(list, biases_name);

    tensor_copy_ip(conv->kernels, loaded_kernels);
    tensor_copy_ip(conv->biases, loaded_biases);

    mga_scratch_release(scratch);
}