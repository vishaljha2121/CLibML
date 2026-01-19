//
// Created by Vishal Jha on 16/01/26.
//
#include "tensor_internal.h"

#include <float.h>
#include <math.h>
#include <string.h>

#if TENSOR_BACKEND == TENSOR_BACKEND_CPU

void _tensor_create_alloc_backend(mg_arena* arena, tensor* out, tensor_shape shape, u64 alloc) {
    out->shape = shape;
    out->alloc = alloc;
    out->data = MGA_PUSH_ZERO_ARRAY(arena, f32, alloc);
}
void _tensor_destroy_backend(tensor* t) {
    // Nothing to do here because data is all on arenas
    UNUSED(t);
}
void _tensor_copy_backend(tensor* out, const tensor* t, u64 size) {
    memcpy(out->data, t->data, sizeof(f32) * size);
}
void _tensor_fill_backend(tensor* tensor, f32 num) {
    tensor_shape shape = tensor->shape;
    u64 size = (u64)shape.width * shape.height * shape.depth;

    f32* data = (f32*)tensor->data;

    for (u64 i = 0; i < size; i++) {
        data[i] = num;
    }
}
tensor_index _tensor_argmax_backend(const tensor* t) {
    f32* data = (f32*)t->data;

    f32 max_num = -FLT_MAX;
    tensor_index max_index = { 0, 0, 0 };

    for (u64 z = 0; z < t->shape.depth; z++) {
        for (u64 y = 0; y < t->shape.height; y++) {
            for (u64 x = 0; x < t->shape.width; x++) {
                if (data[x + y * t->shape.width + z * t->shape.width * t->shape.height] > max_num) {
                    max_num = data[x + y * t->shape.width + z * t->shape.width * t->shape.height];
                    max_index = (tensor_index){ x, y, z };
                }
            }
        }
    }

    return max_index;
}
b32 _tensor_is_zero(const tensor* t) {
    b32 is_zero = true;

    f32* data = (f32*)t->data;

    u64 size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    for (u64 i = 0; i < size; i++) {
        if (data[i] != 0.0f) {
            is_zero = false;
            break;
        }
    }

    return is_zero;

}
void _tensor_2d_view_backend(tensor* out, const tensor* tensor, u32 z) {
    out->shape = (tensor_shape) {
        .width = tensor->shape.width,
        .height = tensor->shape.height,
        .depth = 1
    };
    out->alloc = (u64)out->shape.width * out->shape.height;

    u64 start_i = (u64)z * tensor->shape.width * tensor->shape.height;

    f32* data = tensor->data;
    out->data = (void*)&data[start_i];
}
// Varients of dot with different transposing
// a_width is after transposing
// lda and ldb are the widths of a and b, before transposing

// Neither are transposed
void _dot_nn(tensor* out, u32 a_width, f32* a_data, u32 lda, f32* b_data, u32 ldb) {
    f32* out_data = (f32*)out->data;

    for (u32 y = 0; y < out->shape.height; y++) {
        for (u32 i = 0; i < a_width; i++) {
            // This does not change throughout the inner loop
            f32 a_elem = a_data[(u64)i + (u64)y * lda];
            for (u32 x = 0; x < out->shape.width; x++) {
                out_data[(u64)x + (u64)y * out->shape.width] += a_elem * b_data[(u64)x + (u64)i * ldb];
            }
        }
    }
}

// b is transposed
void _dot_nt(tensor* out, u32 a_width, f32* a_data, u32 lda, f32* b_data, u32 ldb) {
    f32* out_data = (f32*)out->data;

    for (u32 y = 0; y < out->shape.height; y++) {
        for (u32 x = 0; x < out->shape.width; x++) {
            f32 sum = 0.0f;
            for (u32 i = 0; i < a_width; i++) {
                sum += a_data[(u64)i + (u64)y * lda] * b_data[(u64)i + (u64)x * ldb];
            }
            out_data[(u64)x + (u64)y * out->shape.width] = sum;
        }
    }
}

// a is transposed
void _dot_tn(tensor* out, u32 a_width, f32* a_data, u32 lda, f32* b_data, u32 ldb) {
    f32* out_data = (f32*)out->data;

    for (u32 y = 0; y < out->shape.height; y++) {
        for (u32 i = 0; i < a_width; i++) {
            f32 a_elem = a_data[(u64)y + (u64)i * lda];
            for (u32 x = 0; x < out->shape.width; x++) {
                out_data[(u64)x + (u64)y * out->shape.width] += a_elem * b_data[(u64)x + (u64)i * ldb];
            }
        }
    }
}

// Both are 
void _dot_tt(tensor* out, u32 a_width, f32* a_data, u32 lda, f32* b_data, u32 ldb) {
    f32* out_data = (f32*)out->data;

    for (u32 y = 0; y < out->shape.height; y++) {
        for (u32 x = 0; x < out->shape.width; x++) {
            f32 sum = 0.0f;
            for (u32 i = 0; i < a_width; i++) {
                 sum += a_data[(u64)y + (u64)i * lda] * b_data[(u64)i + (u64)x * ldb];
            }
            out_data[(u64)x + (u64)y * out->shape.width] += sum;
        }
    }
}


// Inputs cannot overlap with output
// Output shape should be set up
// TODO: remove
#include <stdio.h>
void _tensor_dot_backend(tensor* out, b32 transpose_a, b32 transpose_b, const tensor* a, const tensor* b) {
    u32 lda = a->shape.width;
    u32 ldb = b->shape.width;

    f32* a_data = (f32*)a->data;
    f32* b_data = (f32*)b->data;

    _tensor_fill_backend(out, 0.0f);

    u32 a_width = transpose_a ? a->shape.height : a->shape.width;

    if (!transpose_a && !transpose_b) {
        _dot_nn(out, a_width, a_data, lda, b_data, ldb);
    } else if (!transpose_a && transpose_b) {
        _dot_nt(out, a_width, a_data, lda, b_data, ldb);
    } else if (transpose_a && !transpose_b) {
        _dot_tn(out, a_width, a_data, lda, b_data, ldb);
    } else {
        _dot_tt(out, a_width, a_data, lda, b_data, ldb);
    }
}
void _tensor_im2col_backend(tensor* out, const tensor* input, u32 kernel_size, u32 stride, u32 padding, u32 x_kernels, u32 y_kernels) {
    tensor_fill(out, 0.0f);

    f32* in_data = (f32*)input->data;
    f32* out_data = (f32*)out->data;

    for (u32 z = 0; z < input->shape.depth; z++) {
        for (u32 k = 0; k < kernel_size * kernel_size; k++) {
            u32 x_off = k % kernel_size;
            u32 y_off = k / kernel_size;

            for (u32 y = 0; y < y_kernels; y++) {
                for (u32 x = 0; x < x_kernels; x++) {
                    u32 in_x = x_off + x * stride - padding;
                    u32 in_y = y_off + y * stride - padding;
                    u64 in_index = ((u64)z * input->shape.height + in_y) * input->shape.width + in_x;

                    u32 out_x = y * x_kernels + x;
                    u32 out_y = (z * kernel_size * kernel_size) + k;
                    u64 out_index = (u64)out_y * out->shape.width + out_x;

                    if (in_x < 0 || in_y < 0 || in_x >= input->shape.width || in_y >= input->shape.height) {
                        out_data[out_index] = 0.0f;
                    } else {
                        out_data[out_index] = in_data[in_index];
                    }
                }
            }
        }
    }
}
void _tensor_col2im_backend(tensor* out, const tensor* input, u32 kernel_size, u32 stride, u32 padding, u32 x_kernels, u32 y_kernels) {
    tensor_fill(out, 0.0f);

    f32* in_data = (f32*)input->data;
    f32* out_data = (f32*)out->data;

    for (u32 z = 0; z < out->shape.depth; z++) {
        for (u32 k = 0; k < kernel_size * kernel_size; k++) {
            u32 x_off = k % kernel_size;
            u32 y_off = k / kernel_size;

            for (u32 y = 0; y < y_kernels; y++) {
                for (u32 x = 0; x < x_kernels; x++) {
                    u32 in_x = y * x_kernels + x;
                    u32 in_y = (z * kernel_size * kernel_size) + k;
                    u64 in_index = (u64)in_y * input->shape.width + in_x;

                    u32 out_x = x_off + x * stride - padding;
                    u32 out_y = y_off + y * stride - padding;
                    u64 out_index = ((u64)z * out->shape.height + out_y) * out->shape.width + out_x;

                    if (out_x >= 0 && out_x < out->shape.width && out_y >= 0 && out_y < out->shape.height) {
                        out_data[out_index] += in_data[in_index];
                    }
                }
            }
        }
    }
}
// Cannot overlap
void _tensor_transpose_backend(tensor* out, const tensor* t) {
    f32* out_data = (f32*)out->data;
    f32* t_data = (f32*)t->data;

    for (u64 x = 0; x < out->shape.width; x++) {
        for (u64 y = 0; y < out->shape.height; y++) {
            out_data[x + y * out->shape.width] = t_data[y + x * t->shape.width];
        }
    }
}
void _tensor_add_backend(tensor* out, const tensor* a, const tensor* b) {
    f32* out_data = (f32*)out->data;
    f32* a_data = (f32*)a->data;
    f32* b_data = (f32*)b->data;

    u64 size = (u64)out->shape.width * out->shape.height * out->shape.depth;

    for (u64 i = 0; i < size; i++) {
        out_data[i] = a_data[i] + b_data[i];
    }
}
void _tensor_sub_backend(tensor* out, const tensor* a, const tensor* b) {
    f32* out_data = (f32*)out->data;
    f32* a_data = (f32*)a->data;
    f32* b_data = (f32*)b->data;

    u64 size = (u64)out->shape.width * out->shape.height * out->shape.depth;

    for (u64 i = 0; i < size; i++) {
        out_data[i] = a_data[i] - b_data[i];
    }
}
void _tensor_component_mul_backend(tensor* out, const tensor* a, const tensor* b) {
    f32* out_data = (f32*)out->data;
    f32* a_data = (f32*)a->data;
    f32* b_data = (f32*)b->data;

    u64 size = (u64)out->shape.width * out->shape.height * out->shape.depth;

    for (u64 i = 0; i < size; i++) {
        out_data[i] = a_data[i] * b_data[i];
    }
}
void _tensor_component_div_backend(tensor* out, const tensor* a, const tensor* b) {
    f32* out_data = (f32*)out->data;
    f32* a_data = (f32*)a->data;
    f32* b_data = (f32*)b->data;

    u64 size = (u64)out->shape.width * out->shape.height * out->shape.depth;

    for (u64 i = 0; i < size; i++) {
        out_data[i] = a_data[i] / b_data[i];
    }
}
void _tensor_add_all_backend(tensor* out, const tensor* t, f32 x) {
    f32* out_data = (f32*)out->data;
    f32* in_data = (f32*)t->data;

    u64 size = (u64)out->shape.width * out->shape.height * out->shape.depth;

    for (u64 i = 0; i < size; i++) {
        out_data[i] = in_data[i] + x;
    }
}
void _tensor_scale_backend(tensor* out, const tensor* t, f32 s) {
    f32* out_data = (f32*)out->data;
    f32* t_data = (f32*)t->data;

    u64 size = (u64)out->shape.width * out->shape.height * out->shape.depth;

    for (u64 i = 0; i < size; i++) {
        out_data[i] = t_data[i] * s;
    }
}
void _tensor_sqrt_backend(tensor* out, const tensor* t) {
    f32* out_data = (f32*)out->data;
    f32* t_data = (f32*)t->data;

    u64 size = (u64)out->shape.width * out->shape.height * out->shape.depth;

    for (u64 i = 0; i < size; i++) {
        out_data[i] = sqrtf(t_data[i]);
    }
}
void _tensor_get_data_backend(f32* out, const tensor* t) {
    f32* t_data = (f32*)t->data;

    if (out == t_data)
        return;

    memcpy(out, t_data, sizeof(f32) * t->shape.width * t->shape.height * t->shape.depth);
}
void _tensor_set_data_backend(tensor* t, f32* data) {
    f32* t_data = (f32*)t->data;

    if (data == t_data)
        return;

    memcpy(t_data, data, sizeof(f32) * t->shape.width * t->shape.height * t->shape.depth);
}

#endif // TENSOR_BACKEND == TENSOR_BACKEND_CPU