//
// Created by Vishal Jha on 16/01/26.
//

#ifndef TENSORNEW_H
#define TENSORNEW_H

#include "base_defs.h"
#include "str.h"
#include "../src/mg/mg_arena.h"

#define TENSOR_BACKEND_CPU 1
#define TENSOR_BACKEND_CUDA 2

#ifndef TENSOR_BACKEND
#   define TENSOR_BACKEND TENSOR_BACKEND_CPU
#endif

/**
 * @brief Shape of `tensor`
 */
typedef struct {
    u32 width;
    u32 height;
    u32 depth;
} tensor_shape;

/**
 * @brief Index into tensor
 *
 * Indexing: `Element[x,y,z] == tensor->data[x + y * width + z * width * height]`
 */
typedef struct {
    u32 x, y, z;
} tensor_index;

/**
 * @brief 3D tensor
 */
typedef struct {
    /**
     * @brief Size of each dim
     *
     * Each should be at least 1 <br>
     * This is ensured in `tensor_create`
     */
    tensor_shape shape;
    /// Number of f32's allocated
    u64 alloc;

    /**
     * @brief Data of tensor
     *
     * Void pointer to support different backends
     */
    void* data;
} tensor;

/**
 * @brief Node in `tensor_list``
 */
typedef struct tensor_node {
    struct tensor_node* next;

    /// Tensor of node
    tensor* tensor;
    /// Name of node
    string8 name;
} tensor_node;

/**
 * @brief List of named tensors
 *
 * You can iterate through the list or
 * get a tensor by name with `tensor_list_get`
 */
typedef struct {
    tensor_node* first;
    tensor_node* last;

    u32 size;
} tensor_list;

/**
 * @brief Whether or not to error if
 *  there is not enough space in out in _ip functions
 */
#ifndef TENSOR_IP_ALLOC_ERRORS
#define TENSOR_IP_ALLOC_ERRORS 1
#endif

/// Returns true if the indices `a` and `b` are equal
b32 tensor_index_eq(tensor_index a, tensor_index b);
/// Returns true if the shapes `a` and `b` are equal
b32 tensor_shape_eq(tensor_shape a, tensor_shape b);

/**
 * @brief Creates a `tensor` and fills it will zero
 *
 * @param arena Arena to allocate `tensor` and data on
 * @param shape Shape of tensor to create.
 *  shape.width MUST be at least 1.
 *  shape.height and shape.depth can be zero.
 *
 * @return The created tensor, filled with zeros
 */
tensor* tensor_create(mg_arena* arena, tensor_shape shape);
/**
 * @brief Creates a tensor with the specified alloc
 *
 * @param arena Arena to allocate `tensor` and data on
 * @param shape Shape of tensor to create.
 *  shape.width MUST be at least 1.
 *  shape.height and shape.depth can be zero.
 * @param alloc Number of `f32`s to allocate.
 *  Must be at least `(u64)shape.width * shape.height * shape.depth`
 *
 * @return The created tensor, filled with zero, and the correct alloc
 */
tensor* tensor_create_alloc(mg_arena* arena, tensor_shape shape, u64 alloc);

// TODO: reword memory useage and age destroy statements

/**
 * @brief Destroys anything on the tensor not stored in an arena
 *
 * This is neccessary because of non-CPU backends
 */
void tensor_destroy(tensor* t);
/**
 * @brief Copies a `tensor`
 *
 * @param arena Arena to create copy on
 * @param tensor Tensor to copy
 * @param keep_alloc Maintain the alloc of the tensor being copied.
 *  If false, the out alloc is based on the shape of the tensor being copied
 *
 * @return The copied tensor
 */
tensor* tensor_copy(mg_arena* arena, const tensor* tensor, b32 keep_alloc);
/**
 * @brief Copies `t` into `out` if `out` is big enough
 *
 * @param out Where `t` gets coppied
 * @param t Tensor to copy
 *
 * @return true if `out` is big enough, `false` otherwise
 */
b32 tensor_copy_ip(tensor* out, const tensor* t);

/// Fills `tensor` with `num`
void tensor_fill(tensor* tensor, f32 num);

/// Returns the index of the maximum element of `t`
tensor_index tensor_argmax(const tensor* t);

/// Returns true if `t` is all zero
b32 tensor_is_zero(const tensor* t);

/**
 * @brief Gets a 2D view from a 3D tensor. DOES NOT COPY THE DATA
 *
 * @param out Output of view
 * @param tensor Tensor you are viewing
 * @param z Index of 2D slice
 */
void tensor_2d_view(tensor* out, const tensor* tensor, u32 z);

/**
 * @brief Computes the dot product of `a` and `b`.
 *
 * `a` and `b` have to be 2D.
 * `a.width` must equal `b.height`
 *
 * @param out Output of dot product. Needs to be big enough (i.e. (b.width, a.height, 1))
 * @param transpose_a Whether or not to transpose a
 * @param transpose_b Whether or not to transpose b
 * @param a First tensor
 * @param b Second tensor
 *
 * @return true if `out` was big enough, false otherwise
 */
b32 tensor_dot_ip(tensor* out, b32 transpose_a, b32 transpose_b, const tensor* a, const tensor* b);
/**
 * @brief Computes the dot product of `a` and `b`. Must be 2D tensors (depth == 1)
 *
 * See `tensor_dot_ip` for more
 */
tensor* tensor_dot(mg_arena* arena, b32 transpose_a, b32 transpose_b, const tensor* a, const tensor* b);

/**
 * @brief Computes the output shape of `tensor_cov`
 *
 * See tensor_conv for more detail
 */
tensor_shape tensor_conv_shape(tensor_shape in_shape, tensor_shape kernel_shape, u32 stride_x, u32 stride_y);

/**
 * @brief Implements the famous `im2col` function. In place version
 *
 * Converts 3d sections of the input image into rows in the output image. <br>
 * `input` and `out` cannot be the same or overlap. <br>
 * Commonly used in convolutional layers to speed up convolutions
 *
 * @param out Output rows
 * @param input Input image
 * @param kernel_size Side length of kernel
 * @param stride Stride of convolution
 * @param padding Padding of image on each side of x and y
 *
 * @return true if `out` is big enough
 */
b32 tensor_im2col_ip(tensor* out, const tensor* input, u32 kernel_size, u32 stride, u32 padding);
/**
 * @brief Implements the `im2col` function
 *
 * See `tensor_im2col_ip` for details
 */
tensor* tensor_im2col(mg_arena* arena, const tensor* input, u32 kernel_size, u32 stride, u32 padding);

/**
 * @brief Implements the famous `col2im` function. In place version
 *
 * Converts rows of input matrix into an image.
 * Used in convolution layers
 *
 * @param out Output image
 * @param input 2D input matrix
 * @param out_shape Shape of output image (width, height, channels)
 * @param kernel_size Side length of kernel
 * @param stride Stride of convolution
 * @param padding Padding of image on each side of x and y
 *
 * @return true if `out` is big enough
 */
b32 tensor_col2im_ip(tensor* out, const tensor* input, tensor_shape out_shape, u32 kernel_size, u32 stride, u32 padding);
/**
 * @brief Implements the famour `col2im` function.
 *
 * See `tensor_col2im` for details
 */
tensor* tensor_col2im(mg_arena* arena, const tensor* input, tensor_shape out_shape, u32 kernel_size, u32 stride, u32 padding);

/**
 * @brief Transposes a 2D tensor in place
 *
 * Must be 2D
 *
 * @return true on success, false otherwise
 */
b32 tensor_transpose_ip(tensor* t);
/**
 * @brief Creates a transposed version of `t`
 */
tensor* tensor_transpose(mg_arena* arena, const tensor* t);

/**
 * @brief Adds `a` and `b` into out
 *
 * @return true if `out` is big enough, false otherwise
 */
b32 tensor_add_ip(tensor* out, const tensor* a, const tensor* b);
/**
 * @brief Subtracts `a` and `b` into out
 *
 * @return true if `out` is big enough, false otherwise
 */
b32 tensor_sub_ip(tensor* out, const tensor* a, const tensor* b);
/**
 * @brief Component multiplies `a` and `b` into out
 *
 * @return true if `out` is big enough, false otherwise
 */
b32 tensor_component_mul_ip(tensor* out, const tensor* a, const tensor* b);
/**
 * @brief Component divides `a` and `b` into out
 *
 * @return true if `out` is big enough, false otherwise
 */
b32 tensor_component_div_ip(tensor* out, const tensor* a, const tensor* b);
/**
 * @brief Adds `x` to every value in `t`
 *
 * @return true if `out` is big enough, false otherwise
 */
b32 tensor_add_all_ip(tensor* out, const tensor* t, f32 x);
/**
 * @brief Scales `t` by `s`
 *
 * @return true if `out` is big enough, false otherwise
 */
b32 tensor_scale_ip(tensor* out, const tensor* t, f32 s);
/**
 * @brief Computes the square root of `t`
 *
 * @return true if `out` is big enough, false otherwise
 */
b32 tensor_sqrt_ip(tensor* out, const tensor* t);

/// Creates a `tensor` that is the sum of `a` and `b`
tensor* tensor_add(mg_arena* arena, const tensor* a, const tensor* b);
/// Creates a `tensor` that is the difference of `a` and `b`
tensor* tensor_sub(mg_arena* arena, const tensor* a, const tensor* b);
/// Creates a `tensor` that is the component product of `a` and `b`
tensor* tensor_component_mul(mg_arena* arena, const tensor* a, const tensor* b);
/// Creates a `tensor` that is the component quotient of `a` and `b`
tensor* tensor_component_div(mg_arena* arena, const tensor* a, const tensor* b);
/// Creates a `tensor` that `x` added to each element of `t`
tensor* tensor_add_all(mg_arena* arena, const tensor* t, f32 x);
/// Creates a `tensor` that is `t` scaled by `s`
tensor* tensor_scale(mg_arena* arena, const tensor* t, f32 s);
/// Creates a `tensor` that is the square root of `t`
tensor* tensor_sqrt(mg_arena* arena, const tensor* t);

/// Returns a copy of the tensor's data
f32* tensor_copy_data(mg_arena* arena, const tensor* t);
/**
 * @brief Copies the tensor's data into out
 *
 * @param out Output of copy, must be big enough to store the data
 * @param t Tensor to get data from
 */
void tensor_get_data(f32* out, const tensor* t);
/**
 * @brief Sets the data of the tensor
 *
 * @param t Tensor to set data of
 * @param data New data of tensor. Must be large enough
 */
void tensor_set_data(tensor* t, f32* data);

/**
 * @brief Pushes a `tensor` and `string8` name to a `tensor_list`
 *  with an existing `tensor_node`
 *
 * @param list List to push to
 * @param tensor Tensor to push onto
 * @param name Name of tensor being pushed
 * @param node Node to push
 */
void tensor_list_push_existing(tensor_list* list, tensor* tensor, string8 name, tensor_node* node);
/**
 * @brief Pushes a `tensor` and `string8` name to a `tensor_list`
 *
 * Does not copy `tensor`
 *
 * @param arena Arena to create node on
 * @param list List to push onto
 * @param tensor Tensor to push
 * @param name Name of tensor being pushed
 */
void tensor_list_push(mg_arena* arena, tensor_list* list, tensor* tensor, string8 name);
/**
 * @brief Gets a `tensor` from a `tensor_list` with a name
 *
 * @param list List to get from
 * @param name Name of tensor to get
 *
 * @return `tensor` corresponding to `name`, or NULL if `name` is not in list
 */
tensor* tensor_list_get(const tensor_list* list, string8 name);

/**
 * @brief Serializes a `tensor_list` to a `string8`
 *
 * Serializes according to the .tst format.
 * See `tensor_list_save` in `tensor.c` for more
 *
 * @param arena Arena to create `string8` on
 * @param list List to serialize
 *
 * @return Serialized list
 */
string8 tensor_list_to_str(mg_arena* arena, const tensor_list* list);
/**
 * @brief Creates a `tensor_list` from a `string8`
 *
 * @param arena Arena to push `tensor`s and `tensor_node`s onto
 * @param str String to load
 *
 * @return List of tensors from the string
 */
tensor_list tensor_list_from_str(mg_arena* arena, string8 str);

/// Returns the .tst file header
string8 tensor_get_tst_header(void);

/**
 * @brief Serializes a `tensor_list` into a file according to the .tst file format
 *
 * See `tensor.c` for more about the format
 *
 * @param list List to save
 * @param file_name Output file. Include file extention in `file_name`
 */
void tensor_list_save(const tensor_list* list, string8 file_name);
/**
 * @brief Loads a `tensor_list` from a file
 *
 * @param arena Arena to push `tensor`s and `tensor_node`s onto
 * @param file_name File to load
 *
 * @return List of tensors from file
 */
tensor_list tensor_list_load(mg_arena* arena, string8 file_name);

#endif //TENSORNEW_H
