//
// Created by Vishal Jha on 16/01/26.
//

#include "../../include/tensorNew.h"
#include "tensor.h"

#include "tensor_internal.h"

#include "../../include/os.h"
#include "../../include/err.h"

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

b32 tensor_index_eq(tensor_index a, tensor_index b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
b32 tensor_shape_eq(tensor_shape a, tensor_shape b) {
    return a.width == b.width && a.height == b.height && a.depth == b.depth;
}

tensor* tensor_create(mg_arena* arena, tensor_shape shape) {
    if (shape.height == 0) { shape.height = 1; }
    if (shape.depth == 0) { shape.depth = 1; }

    u64 alloc = (u64)shape.width * shape.height * shape.depth;
    return tensor_create_alloc(arena, shape, alloc);
}
tensor* tensor_create_alloc(mg_arena* arena, tensor_shape shape, u64 alloc) {
    if (shape.width == 0) {
        ERR(ERR_BAD_SHAPE, "Cannot create tensor of width 0");
        return NULL;
    }

    if (shape.height == 0) { shape.height = 1; }
    if (shape.depth == 0) { shape.depth = 1; }

    u64 min_alloc = (u64)shape.width * shape.height * shape.depth;
    if (alloc < min_alloc) {
        ERR(ERR_INVALID_INPUT, "Cannot create tensor, alloc is too small");

        return NULL;
    }

    tensor* out = MGA_PUSH_STRUCT(arena, tensor);

    _tensor_create_alloc_backend(arena, out, shape, alloc);

    return out;
}
void tensor_destroy(tensor* t) {
    _tensor_destroy_backend(t);
}
tensor* tensor_copy(mg_arena* arena, const tensor* t, b32 keep_alloc) {
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot copy NULL tensor");

        return NULL;
    }

    tensor_shape shape = t->shape;
    u64 alloc = keep_alloc ? t->alloc : ((u64)shape.width * shape.height * shape.depth);

    tensor* out = tensor_create_alloc(arena, shape, alloc);

    _tensor_copy_backend(out, t, out->alloc);

    return out;
}
b32 tensor_copy_ip(tensor* out, const tensor* t) {
    if (out == NULL || t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot copy tensor: out and/or tensor is NULL");

        return false;
    }

    u64 size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < size) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot copy tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = t->shape;
    if (out->data != t->data) {
        _tensor_copy_backend(out, t, size);
    }

    return true;
}

void tensor_fill(tensor* tensor, f32 num) {
    if (tensor == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot fill NULL tensor");
    }

    _tensor_fill_backend(tensor, num);
}

tensor_index tensor_argmax(const tensor* t) {
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot get argmax of NULL tensor");

        return (tensor_index){ 0 };
    }

    return _tensor_argmax_backend(t);
}

b32 tensor_is_zero(const tensor* t) {
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot test if NULL tensor is zero");

        return false;
    }

    return _tensor_is_zero(t);
}
void tensor_2d_view(tensor* out, const tensor* tensor, u32 z) {
    if (out == NULL || tensor == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot create 2d view will NULL tensor(s)");

        return;
    }

    _tensor_2d_view_backend(out, tensor, z);
}

b32 tensor_dot_ip(tensor* out, b32 transpose_a, b32 transpose_b, const tensor* a, const tensor* b) {
    if (out == NULL || a == NULL || b == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot dot with NULL tensor(s)");

        return false;
    }

    if (a->shape.depth != 1 || b->shape.depth != 1) {
        ERR(ERR_BAD_SHAPE, "Cannot dot tensor in 3 dimensions");

        return false;
    }

    tensor_shape a_shape = a->shape;
    tensor_shape b_shape = b->shape;
    if (transpose_a) {
        u32 tmp = a_shape.width;
        a_shape.width = a_shape.height;
        a_shape.height = tmp;
    }
    if (transpose_b) {
        u32 tmp = b_shape.width;
        b_shape.width = b_shape.height;
        b_shape.height = tmp;
    }

    if (a_shape.width != b_shape.height) {
        ERR(ERR_BAD_SHAPE, "Cannot dot tensor: shapes do not align");

        return false;
    }

    tensor_shape out_shape = {
        .width = b_shape.width,
        .height = a_shape.height,
        .depth = 1
    };
    u64 data_size = (u64)out_shape.width * out_shape.height;

    if (out->alloc < data_size) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot dot tensor: not enough space in out");
        #endif

        return false;
    }

    // Casting out const is not the best idea,
    // but I think its fine here
    tensor* real_a = (tensor*)a;
    tensor* real_b = (tensor*)b;
    u32 copied = 0;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    if (a == out) {
        real_a = tensor_copy(scratch.arena, a, false);
        copied |= 0b1;
    }
    if (b == out) {
        real_b = tensor_copy(scratch.arena, b, false);
        copied |= 0b10;
    }

    out->shape = out_shape;

    _tensor_dot_backend(out, transpose_a, transpose_b, real_a, real_b);

    if (copied & 0b1)
        tensor_destroy(real_a);
    if (copied & 0b10)
        tensor_destroy(real_b);

    mga_scratch_release(scratch);

    return true;
}
tensor* tensor_dot(mg_arena* arena, b32 transpose_a, b32 transpose_b, const tensor* a, const tensor* b) {
    if (a == NULL || b == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot dot with NULL tensor(s)");

        return NULL;
    }

    tensor_shape shape = {
        transpose_b ? b->shape.height : b->shape.width,
        transpose_a ? a->shape.width : a->shape.height,
        1
    };

    tensor* out = tensor_create(arena, shape);

    tensor_dot_ip(out, transpose_a, transpose_b, a, b);

    return out;
}

tensor_shape tensor_conv_shape(tensor_shape in_shape, tensor_shape kernel_shape, u32 stride_x, u32 stride_y) {
    tensor_shape out_shape = { 0, 0, 1 };

    if (stride_x == 0 || stride_y == 0) {
        ERR(ERR_INVALID_INPUT, "Cannot create conv shape with strides of zero");

        return out_shape;
    }

    out_shape.width = (in_shape.width - kernel_shape.width) / stride_x + 1;
    out_shape.height = (in_shape.height - kernel_shape.height) / stride_y + 1;

    return out_shape;
}
b32 tensor_im2col_ip(tensor* out, const tensor* input, u32 kernel_size, u32 stride, u32 padding) {
    if (out == NULL || input == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot im2col with NULL tensor(s)");

        return false;
    }

    if (stride == 0) {
        ERR(ERR_INVALID_INPUT, "Cannot convert image to cols with stride of zero");

        return false;
    }
    if (out->data == input->data) {
        ERR(ERR_INVALID_INPUT, "Cannot convert image to cols when out and input overlap");

        return false;
    }

    // Number of kernels that fit in input on the x and y axes
    u32 x_kernels = (input->shape.width + padding * 2 - kernel_size) / stride + 1;
    u32 y_kernels = (input->shape.height + padding * 2 - kernel_size) / stride + 1;

    tensor_shape shape = {
        x_kernels * y_kernels,
        input->shape.depth * kernel_size * kernel_size,
        1
    };

    u64 out_alloc = (u64)shape.width * shape.height * shape.depth;
    if (out->alloc < out_alloc) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot convert image to cols: not enough space in out");
        #endif

        return false;
    }

    out->shape = shape;

    _tensor_im2col_backend(out, input, kernel_size, stride, padding, x_kernels, y_kernels);

    return true;
}
tensor* tensor_im2col(mg_arena* arena, const tensor* input, u32 kernel_size, u32 stride, u32 padding) {
    if (input == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot convert NULL tensor to cols");

        return NULL;
    }

    if (stride == 0) {
        ERR(ERR_INVALID_INPUT, "Cannot convert image to cols with stride of zero");

        return false;
    }

    // Number of kernels that fit in input on the x and y axes
    u32 x_kernels = (input->shape.width + padding * 2 - kernel_size) / stride + 1;
    u32 y_kernels = (input->shape.height + padding * 2 - kernel_size) / stride + 1;

    tensor_shape shape = {
        x_kernels * y_kernels,
        input->shape.depth * kernel_size * kernel_size,
        1
    };

    mga_temp maybe_temp = mga_temp_begin(arena);
    tensor* out = tensor_create(arena, shape);

    if (!tensor_im2col_ip(out, input, kernel_size, stride, padding)) {
        mga_temp_end(maybe_temp);

        out = NULL;
    }

    return out;
}

b32 tensor_col2im_ip(tensor* out, const tensor* input, tensor_shape out_shape, u32 kernel_size, u32 stride, u32 padding) {
   if (out == NULL || input == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot col2im with NULL tensor(s)");

        return false;
    }

    if (stride == 0) {
        ERR(ERR_INVALID_INPUT, "Cannot convert cols to image with stride of zero");

        return false;
    }
    if (out->data == input->data) {
        ERR(ERR_INVALID_INPUT, "Cannot convert cols to image when out and input overlap");

        return false;
    }

    u64 out_alloc = (u64)out_shape.width * out_shape.height * out_shape.depth;
    if (out->alloc < out_alloc) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot convert cols to image: not enough space in out");
        #endif

        return false;
    }
    out->shape = out_shape;

    u32 x_kernels = (out_shape.width + padding * 2 - kernel_size) / stride + 1;
    u32 y_kernels = (out_shape.height + padding * 2 - kernel_size) / stride + 1;

    _tensor_col2im_backend(out, input, kernel_size, stride, padding, x_kernels, y_kernels);

    return true;
}
tensor* tensor_col2im(mg_arena* arena, const tensor* input, tensor_shape out_shape, u32 kernel_size, u32 stride, u32 padding) {
    if (input == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot convert NULL tensor to image");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);
    tensor* out = tensor_create(arena, out_shape);

    if (!tensor_col2im_ip(out, input, out_shape, kernel_size, stride, padding)) {
        mga_temp_end(maybe_temp);

        out = NULL;
    }

    return out;
}

b32 tensor_transpose_ip(tensor* t) {
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot transpose NULL tensor");

        return false;
    }

    if (t->shape.depth != 1) {
        ERR(ERR_BAD_SHAPE, "Cannot transpose tensor with depth");

        return false;
    }

    u32 temp_width = t->shape.width;
    t->shape.width = t->shape.height;
    t->shape.height = temp_width;

    // If it is 1d, you do not need to move around the numbers
    if (t->shape.width == 1 || t->shape.height == 1) {
        return true;
    }

    // Creating temporary copy of data
    mga_temp scratch = mga_scratch_get(NULL, 0);

    tensor* orig = tensor_copy(scratch.arena, t, false);
    orig->shape.width = t->shape.height;
    orig->shape.height = t->shape.width;

    _tensor_transpose_backend(t, orig);

    tensor_destroy(orig);

    mga_scratch_release(scratch);

    return true;
}
tensor* tensor_transpose(mg_arena* arena, const tensor* t) {
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot transpose NULL tensor");

        return NULL;
    }

    if (t->shape.depth != 1) {
        ERR(ERR_BAD_SHAPE, "Cannot transpose tensor with depth");

        return NULL;
    }

    tensor* out = tensor_create(arena, (tensor_shape){ t->shape.height, t->shape.width, 1 });

    _tensor_transpose_backend(out, t);

    return out;
}

b32 tensor_add_ip(tensor* out, const tensor* a, const tensor* b) {
    if (out == NULL || a == NULL || b == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot add NULL tensor(s)");

        return false;
    }
    if (!tensor_shape_eq(a->shape, b->shape)) {
        ERR(ERR_BAD_SHAPE, "Cannot add tensor: shapes do not align");

        return false;
    }

    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot add tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = a->shape;

    _tensor_add_backend(out, a, b);

    return true;
}
b32 tensor_sub_ip(tensor* out, const tensor* a, const tensor* b) {
    if (out == NULL || a == NULL || b == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot subtract NULL tensor(s)");

        return false;
    }
    if (!tensor_shape_eq(a->shape, b->shape)) {
        ERR(ERR_BAD_SHAPE, "Cannot subtract tensor: shapes do not align");

        return false;
    }

    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot subtract tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = a->shape;

    _tensor_sub_backend(out, a, b);

    return true;
}
b32 tensor_component_mul_ip(tensor* out, const tensor* a, const tensor* b) {
    if (out == NULL || a == NULL || b == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot component multiply NULL tensor(s)");

        return false;
    }
    if (!tensor_shape_eq(a->shape, b->shape)) {
        ERR(ERR_BAD_SHAPE, "Cannot multiply tensor: shapes do not align");

        return false;
    }

    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot multiply tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = a->shape;

    _tensor_component_mul_backend(out, a, b);

    return true;
}
b32 tensor_component_div_ip(tensor* out, const tensor* a, const tensor* b) {
    if (out == NULL || a == NULL || b == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot component divide NULL tensor(s)");

        return false;
    }
    if (!tensor_shape_eq(a->shape, b->shape)) {
        ERR(ERR_BAD_SHAPE, "Cannot divide tensor: shapes do not align");

        return false;
    }

    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot divide tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = a->shape;

    _tensor_component_div_backend(out, a, b);

    return true;
}
b32 tensor_add_all_ip(tensor* out, const tensor* t, f32 x) {
    if (out == NULL || t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot add all with NULL tensor(s)");

        return false;
    }

    u64 data_size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot add all to tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = t->shape;

    _tensor_add_all_backend(out, t, x);

    return true;
}
b32 tensor_scale_ip(tensor* out, const tensor* t, f32 s) {
    if (out == NULL || t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot scale NULL tensor(s)");

        return false;
    }
    u64 data_size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot scale tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = t->shape;

    _tensor_scale_backend(out, t, s);

    return true;
}
b32 tensor_sqrt_ip(tensor* out, const tensor* t) {
    if (out == NULL || t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot sqrt NULL tensor(s)");

        return false;
    }
    u64 data_size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_IP_ALLOC_ERRORS
        ERR(ERR_ALLOC_SIZE, "Cannot sqrt tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = t->shape;

    _tensor_sqrt_backend(out, t);

    return true;
}

tensor* tensor_add(mg_arena* arena, const tensor* a, const tensor* b) {
    if (a == NULL || b == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot add NULL tensor(s)");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, a->shape);

    if (!tensor_add_ip(out, a, b)) {
        tensor_destroy(out);
        mga_temp_end(maybe_temp);

        out = NULL;
    }

    return out;
}
tensor* tensor_sub(mg_arena* arena, const tensor* a, const tensor* b) {
    if (a == NULL || b == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot sub NULL tensor(s)");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, a->shape);

    if (!tensor_sub_ip(out, a, b)) {
        tensor_destroy(out);
        mga_temp_end(maybe_temp);

        out = NULL;
    }

    return out;
}
tensor* tensor_component_mul(mg_arena* arena, const tensor* a, const tensor* b) {
    if (a == NULL || b == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot component multiply NULL tensor(s)");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, a->shape);

    if (!tensor_component_mul_ip(out, a, b)) {
        tensor_destroy(out);
        mga_temp_end(maybe_temp);

        out = NULL;
    }

    return out;
}
tensor* tensor_component_div(mg_arena* arena, const tensor* a, const tensor* b) {
    if (a == NULL || b == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot component divide NULL tensor(s)");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, a->shape);

    if (!tensor_component_div_ip(out, a, b)) {
        tensor_destroy(out);
        mga_temp_end(maybe_temp);

        out = NULL;
    }

    return out;
}
tensor* tensor_add_all(mg_arena* arena, const tensor* t, f32 x) {
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot add all to NULL tensor");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, t->shape);

    if (!tensor_add_all_ip(out, t, x)) {
        tensor_destroy(out);
        mga_temp_end(maybe_temp);

        out = NULL;
    }

    return out;

}
tensor* tensor_scale(mg_arena* arena, const tensor* t, f32 s) {
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot scale NULL tensor");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, t->shape);

    if (!tensor_scale_ip(out, t, s)) {
        tensor_destroy(out);
        mga_temp_end(maybe_temp);

        out = NULL;
    }

    return out;
}
tensor* tensor_sqrt(mg_arena* arena, const tensor* t) {
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot sqrt NULL tensor");

        return NULL;
    }

    tensor* out = tensor_create(arena, t->shape);

    tensor_sqrt_ip(out, t);

    return out;
}

f32* tensor_copy_data(mg_arena* arena, const tensor* t) {
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot copy data from NULL tensor");

        return NULL;
    }


    u64 size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    f32* out = MGA_PUSH_ARRAY(arena, f32, size);

    _tensor_get_data_backend(out, t);

    return out;
}
void tensor_get_data(f32* out, const tensor* t) {
    if (out == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot get tensor data with NULL out pointer");

        return;
    }
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot get data of NULL tensor");

        return;
    }

    _tensor_get_data_backend(out, t);
}
void tensor_set_data(tensor* t, f32* data) {
    if (t == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot set data of NULL tensor");

        return;
    }
    if (data == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot set tensor data with NULL data");

        return;
    }

    _tensor_set_data_backend(t, data);
}

void tensor_list_push_existing(tensor_list* list, tensor* tensor, string8 name, tensor_node* node) {
    if (list == NULL || tensor == NULL || node == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot push node to tensor list: list, node, or tensor is NULL");

        return;
    }

    node->tensor = tensor;
    node->name = name;

    SLL_PUSH_BACK(list->first, list->last, node);

    list->size++;
}
void tensor_list_push(mg_arena* arena, tensor_list* list, tensor* tensor, string8 name) {
    if (list == NULL || tensor == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot push tensor to list: list or tensor is NULL");

        return;
    }

    tensor_node* node = MGA_PUSH_ZERO_STRUCT(arena, tensor_node);
    tensor_list_push_existing(list, tensor, name, node);
}
tensor* tensor_list_get(const tensor_list* list, string8 name) {
    if (list == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot get tensor from NULL list");

        return NULL;
    }

    tensor* out = NULL;

    for (tensor_node* node = list->first; node != NULL; node = node->next) {
        if (str8_equals(node->name, name)) {
            return node->tensor;
        }
    }

    return out;
}

/*
TODO: Figure out how to make it endian independent

File Format (*.tst):
- Header "tensors"
- u32 num_tensors
- List of tensors
    - Name
        - u64 size
        - u8* str (of length size)
    - tensor
        - u32 width, height, depth
        - f32* data (of length width*height*depth)
*/

static const string8 _tst_header = {
    .size = 10,
    .str = (u8*)"tensors"
};

string8 tensor_get_tst_header(void) {
    return _tst_header;
}

#define _WRITE_DATA(size, data) do { \
        memcpy(str_buf_ptr, (data), (size)); \
        str_buf_ptr += (size); \
    } while (0)

string8 tensor_list_to_str(mg_arena* arena, const tensor_list* list) {
    if (list == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot convert NULL tensor list to string");

        return (string8){ 0 };
    }

    u64 str_size = 0;
    u8* str_buf = NULL;

    str_size += _tst_header.size;
    str_size += sizeof(u32); // for number of tensors

    for (tensor_node* node = list->first; node != NULL; node = node->next) {
        str_size += sizeof(u64); // for str size
        str_size += node->name.size; // for str

        tensor_shape shape = node->tensor->shape;

        str_size += sizeof(u32) * 3; // for width, height, and depth
        str_size += (u64)shape.width * shape.height * shape.depth * sizeof(f32); // for data
    }

    str_buf = MGA_PUSH_ARRAY(arena, u8, str_size);
    u8* str_buf_ptr = str_buf;

    _WRITE_DATA(_tst_header.size, _tst_header.str);
    _WRITE_DATA(sizeof(u32), &list->size);

    for (tensor_node* node = list->first; node != NULL; node = node->next) {
        _WRITE_DATA(sizeof(u64), &node->name.size);
        _WRITE_DATA(node->name.size, node->name.str);

        tensor_shape shape = node->tensor->shape;

        _WRITE_DATA(sizeof(u32), &shape.width);
        _WRITE_DATA(sizeof(u32), &shape.height);
        _WRITE_DATA(sizeof(u32), &shape.depth);

        u64 data_size = (u64)shape.width * shape.height * shape.depth * sizeof(f32);

        mga_temp scratch = mga_scratch_get(NULL, 0);
        f32* data = MGA_PUSH_ARRAY(scratch.arena, f32, data_size);

        _tensor_get_data_backend(data, node->tensor);

        _WRITE_DATA(data_size, data);

        mga_scratch_release(scratch);
    }

    if (str_size != (u64)(str_buf_ptr - str_buf)) {
        ERR(ERR_GENERAL, "Cannnot create tensor string: buffer was not filled");

        return (string8){ 0 };
    }

    return (string8){ .str = str_buf, .size = str_size };
}

#define _READ_DATA(data_size, data) do { \
        if (pos + (data_size) > str.size) { \
            memset((data), 0, (data_size)); \
        } else { \
            memcpy((data), &str.str[pos], (data_size)); \
            pos += (data_size); \
        } \
    } while (0)

tensor_list tensor_list_from_str(mg_arena* arena, string8 str) {
    if (!str8_equals(_tst_header, str8_substr(str, 0, _tst_header.size))) {
        ERR(ERR_PARSE, "Cannot read tensor string: tensor header not found");

        return (tensor_list){ 0 };
    }

    u64 pos = _tst_header.size;

    tensor_list out = { 0 };

    u32 size = 0;
    _READ_DATA(sizeof(u32), &size);

    for (u32 i = 0; i < size; i++) {
        u64 name_size = 0;
        _READ_DATA(sizeof(u64), &name_size);

        string8 name = {
            .size = name_size,
            .str = MGA_PUSH_ZERO_ARRAY(arena, u8, name_size)
        };

        _READ_DATA(name_size, name.str);

        u32 width = 0;
        u32 height = 0;
        u32 depth = 0;

        _READ_DATA(sizeof(u32), &width);
        _READ_DATA(sizeof(u32), &height);
        _READ_DATA(sizeof(u32), &depth);

        tensor* tensor = tensor_create(arena, (tensor_shape){ width, height, depth });
        u64 data_size = (u64)width * height * depth * sizeof(f32);

#if TENSOR_BACKEND == TENSOR_BACKEND_CPU
        _READ_DATA(data_size, tensor->data);
#else
        mga_temp scratch = mga_scratch_get(NULL, 0);

        f32* data = MGA_PUSH_ARRAY(scratch.arena, f32, data_size);
        _READ_DATA(data_size, data);
        _tensor_set_data_backend(tensor, data);

        mga_scratch_release(scratch);
#endif

        tensor_list_push(arena, &out, tensor, name);
    }

    if (pos > str.size) {
        ERR(ERR_PARSE, "Could not load all tensors: cannot read outisde string bounds");
    }

    return out;
}

void tensor_list_save(const tensor_list* list, string8 file_name) {
    if (list == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot save NULL list");

        return;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    string8 file_str = tensor_list_to_str(scratch.arena, list);

    if (file_str.size == 0) {
        ERR(ERR_GENERAL, "Cannnot write tensor file: string was not created");
    } else {
        string8_list output_list = { 0 };
        str8_list_push(scratch.arena, &output_list, file_str);

        file_write(file_name, output_list);
    }

    mga_scratch_release(scratch);
}

tensor_list tensor_list_load(mg_arena* arena, string8 file_name) {
    mga_temp scratch = mga_scratch_get(&arena, 1);

    string8 file = file_read(scratch.arena, file_name);
    if (file.size == 0) {
        ERR(ERR_IO, "Cannot load tensors: failed to read file");

        mga_scratch_release(scratch);
        return (tensor_list){ 0 };
    }

    tensor_list out = tensor_list_from_str(arena, file);

    mga_scratch_release(scratch);

    return out;
}