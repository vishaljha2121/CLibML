//
// Created by Vishal Jha on 16/01/26.
//

#ifndef TENSOR_INTERNAL_H
#define TENSOR_INTERNAL_H

#include "../../include/tensorNew.h"

// All of these are implemented by different backends
// All of these assume arguments are valid

void _tensor_create_alloc_backend(mg_arena* arena, tensor* out, tensor_shape shape, u64 alloc);
void _tensor_destroy_backend(tensor* t);
void _tensor_copy_backend(tensor* out, const tensor* t, u64 size);
void _tensor_fill_backend(tensor* tensor, f32 num);
tensor_index _tensor_argmax_backend(const tensor* t);
b32 _tensor_is_zero(const tensor* t);
void _tensor_2d_view_backend(tensor* out, const tensor* tensor, u32 z);
// Inputs cannot overlap with output
void _tensor_dot_backend(tensor* out, b32 transpose_a, b32 transpose_b, const tensor* a, const tensor* b);
void _tensor_im2col_backend(tensor* out, const tensor* input, u32 kernel_size, u32 stride, u32 padding, u32 x_kernels, u32 y_kernels);
void _tensor_col2im_backend(tensor* out, const tensor* input, u32 kernel_size, u32 stride, u32 padding, u32 x_kernels, u32 y_kernels);
// Cannot overlap
void _tensor_transpose_backend(tensor* out, const tensor* t);
void _tensor_add_backend(tensor* out, const tensor* a, const tensor* b);
void _tensor_sub_backend(tensor* out, const tensor* a, const tensor* b);
void _tensor_component_mul_backend(tensor* out, const tensor* a, const tensor* b);
void _tensor_component_div_backend(tensor* out, const tensor* a, const tensor* b);
void _tensor_add_all_backend(tensor* out, const tensor* t, f32 x);
void _tensor_scale_backend(tensor* out, const tensor* t, f32 s);
void _tensor_sqrt_backend(tensor* out, const tensor* t);
void _tensor_get_data_backend(f32* out, const tensor* t);
void _tensor_set_data_backend(tensor* t, f32* data);

#endif // TENSOR_INTERNAL_H