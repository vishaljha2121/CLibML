//
// Created by Vishal Jha on 16/01/26.
//

/**
 * @file layers.h
 * @brief Neural network layers
 */

#ifndef LAYERS_H
#define LAYERS_H

#include "../src/mg//mg_arena.h"

#include "base_defs.h"
#include "str.h"
#include "tensorNew.h"
#include "optimizers.h"

/**
 * @brief Type of layer
 */
typedef enum {
    /// Does not do anything, should not be used
    LAYER_NULL = 0,
    /// Must be first layer of neural network
    LAYER_INPUT,
    /// Reshapes input of layer and delta of backprop
    LAYER_RESHAPE,

    // TODO: this
    /// Custom layer, user provided functions
    //LAYER_CUSTOM,

    /// Fully connected layer
    LAYER_DENSE,
    /// Applies activation function to input and multiplies activation gradient with delta
    LAYER_ACTIVATION,
    /// Randomly turns off some neurons during training
    LAYER_DROPOUT,
    /// Reshapes input to 1D
    LAYER_FLATTEN,
    /// Pools all 2D slices of a 3D input
    LAYER_POOLING_2D,
    /// 2D convolution layer
    LAYER_CONV_2D,

    /// Layer normalization
    LAYER_NORM,

    /// Number of layers
    LAYER_COUNT
} layer_type;

/**
 * @brief Parameter initialization types
 *
 * Layers will initialize trainable parameters with one of these types <br>
 * See `param_init` for more 
 */
typedef enum {
    /// Does nothing to param
    PARAM_INIT_NULL = 0,

    /// Fills param with zeors
    PARAM_INIT_ZEROS,
    /// Fills param with zeors
    PARAM_INIT_ONES,

    /// Xavier Glorot uniform
    PARAM_INIT_XAVIER_UNIFORM,
    /// Xavier Glorot normal
    PARAM_INIT_XAVIER_NORMAL,

    /// He/Kaiming uniform
    PARAM_INIT_HE_UNIFORM,
    /// He/Kaiming normal
    PARAM_INIT_HE_NORMAL,

    /// Number of param init types
    PARAM_INIT_COUNT
} param_init_type;

/**
 * @brief Activation types for activation layer
 */
typedef enum {
    /// Does nothing to inputs
    ACTIVATION_NULL = 0,
    /// Linear function
    ACTIVATION_LINEAR,
    /// Sigmoid
    ACTIVATION_SIGMOID,
    /// Tanh
    ACTIVATION_TANH,
    /// Relu
    ACTIVATION_RELU,
    /// Leaky relu with leaky value of 0.01
    ACTIVATION_LEAKY_RELU,
    /// Softmax
    ACTIVATION_SOFTMAX,

    /// Number of activation functions
    ACTIVATION_COUNT
} layer_activation_type;

/**
 * @brief Pooling types for pooling layer
 */
typedef enum {
    /// Does nothing
    POOLING_NULL = 0,
    /// Max pooling
    POOLING_MAX,
    /// Average pooling
    POOLING_AVG,

    /// NUmber of pooling types
    POOLING_COUNT
} layer_pooling_type;

struct layer;
struct layer_desc;
struct layers_cache;

/**
 * @brief Layer creation function type
 * 
 * @param arena Arena to initialize layer parameters
 * @param out Layer to initialize. out->shape needs to be set
 * @param desc Layer description
 * @param prev_shape Shape of previous layer in network
 */
typedef void (layer_create_func)(mg_arena* arena, struct layer* out, const struct layer_desc* desc, tensor_shape prev_shape);
/**
 * @brief Layer feedforward function type
 *
 * @param l Layer
 * @param in_out Input and output tensor. The layer should put
 *  the output of the feedforward into this tensor at the end
 * @param cache Tensor cache for backpropagation. This will be NULL sometimes.
 *  If the layer needs to cache values for backpropagation, they should be stored in the cache.
 *  This is necessary because the training takes place on multiple threads
 */
typedef void (layer_feedforward_func)(struct layer* l, tensor* in_out, struct layers_cache* cache);
/**
 * @brief Backpropagation function type
 *
 * @param l Layer
 * @param delta Current delta in backpropagation. This should be updated
 *  by the layer if necessary
 * @param cache Tensor cache, will not be NULL
 */
typedef void (layer_backprop_func)(struct layer* l, tensor* delta, struct layers_cache* cache);
/**
 * @brief Apply changes function type
 *
 * Layers should apply any changes accumulated during training in this function.
 * Changes should be stored using a `param_change`
 *
 * @param l Layer
 * @param optim Training optimizer, passed into the function `param_change_apply`
 */
typedef void (layer_apply_changes_func)(struct layer* l, const optimizer* optim);
/**
 * @beirf Delete function type
 *
 * @param l Layer to delete
 */
typedef void (layer_delete_func)(struct layer* l);
/**
 * @brief Save layer function type
 *
 * Layers should only save trainable parameters. 
 * See layers_dense.c for an example
 *
 * @param arena Arena to use to push onto tensor list
 * @param l Layer to save
 * @param list Tensor list to save parameters to
 * @param index Index of the current layer.
 *  Should be used to make the name of the tensor in the list unique
 */
typedef void (layer_save_func)(mg_arena* arena, struct layer* l, tensor_list* list, u32 index);
/**
 * @brief Load layer function
 *
 * Loads any trainable parameters in the layer. 
 * See layers_dense.c for an example
 *
 * @param l Layer to load to
 * @param list List to load parameters from
 * @param index Index of layer
 */
typedef void (layer_load_func)(struct layer* l, const tensor_list* list, u32 index);

/// Input layer description 
typedef struct {
    /**
     * @brief Shape of input layer
     *
     * Will reshape the neural network input to this shape
     */
    tensor_shape shape;
} layer_input_desc;

/// Reshape layer description
typedef struct {
    /**
     * @brief Shape of layer output
     *
     * Will reshape layer input to shape and backprop delta to input shape
     */
    tensor_shape shape;
} layer_reshape_desc;

// TODO: include save and load? custom layer descs?????
/**
 * @brief Custom layer description
 */
typedef struct {
    /// Creation function or NULL
    layer_create_func* create;
    /// Feedforward function or NULL
    layer_feedforward_func* feedforward;
    /// Backprop function or NULL
    layer_backprop_func* backprop;
    /// Apply changes function or NULL
    layer_apply_changes_func* apply_changes;
    /// Delete function or NULL
    layer_delete_func* delete;
    /// Parameters save function or NULL
    layer_save_func* save;
    /// Parameters load or NULL
    layer_load_func* load;
} layer_custom_desc;

/**
 * @brief Dense layer description
 *
 * Output shape is (`size`, 1, 1)
 */
typedef struct {
    /// Output size of layer
    u32 size;

    /**
     * @brief Initialization type for bias
     *
     * Defaults to PARAM_INIT_ZEROS
     */
    param_init_type bias_init;

    /**
     * @brief Initialization type for weight
     *
     * Defaults to PARAM_INIT_XAVIER_UNIFORM
     */
    param_init_type weight_init;
} layer_dense_desc;

/**
 * @brief Activation layer description
 *
 * Activation layers maintain the previous layer's shape
 */
typedef struct {
    /**
     * @brief Which activation function to use
     *
     * Defaults to ACTIVATION_RELU
     */
    layer_activation_type type;
} layer_activation_desc;

/**
 * @brief Dropout layer description
 *
 * Random dropout is only applied during training <br>
 * Dropout layers maintain the previous layer's shape
 */
typedef struct {
    /// Keeprate for dropout
    f32 keep_rate;
} layer_dropout_desc;

/**
 * @brief 2D Pooling layer description
 */
typedef struct {
    /**
     * @brief Size of pooling
     *
     * depth of `pool_size` is ignored
     */
    tensor_shape pool_size;

    /**
     * @brief Type of pooling to use
     *
     * Defaults to POOLING_MAX
     */
    layer_pooling_type type;
} layer_pooling_2d_desc;

/**
 * @brief 2D Convolutional layer description
 */
typedef struct {
    /**
     * @brief Number of output filters
     *
     * Depth of output shape will equal `num_filters`
     */
    u32 num_filters;

    /**
     * @brief Side length of kernel for convolution operation
     */
    u32 kernel_size;

    /**
     * @brief Adds padding to input before the convolution operation
     *
     * The output size will equal the input size if
     * the strides are 1 and padding is true
     */
    b32 padding;

    /// Stride for convolution. Defaults to 1
    u32 stride;

    /**
     * @brief Initialization type for kernels
     *
     * Defaults to PARAM_INIT_HE_NORMAL
     */
    param_init_type kernels_init;

    /**
     * @brief Initialization type for biases
     *
     * Defaults to PARAM_INIT_ZEROS
     */

    param_init_type biases_init;
} layer_conv_2d_desc;

/**
 * @brief Layer normalization
 */
typedef struct {
    /**
     * @brief Parameter for numerical stability
     *
     * out = (in - mean) / sqrt(std_dev**2 + epsilon)
     */
    f32 epsilon;
} layer_norm_desc;

/**
 * @brief Full layer description
 */ 
typedef struct layer_desc {
    /**
     * @brief Type of layer
     *
     * Used to determine which member of the union is used
     */
    layer_type type;
    /**
     * @brief Used to determine if layer should be created for training
     *
     * Training mode uses more memory, but is necessary for training the network. <br>
     * Only use it when training the network
     */
    b32 training_mode;

    union {
        /// Input desc
        layer_input_desc input;
        /// Reshape desc
        layer_reshape_desc reshape;
        /// Dense desc
        layer_dense_desc dense;
        /// Activation desc
        layer_activation_desc activation;
        /// Dropout desc
        layer_dropout_desc dropout;
        /// Pooling2D desc
        layer_pooling_2d_desc pooling_2d;
        /// Convolutional2D desc
        layer_conv_2d_desc conv_2d;
        /// Layer normalization desc
        layer_norm_desc norm;
    };
} layer_desc;

/**
 * @brief Layer structure
 *
 * Defined in layers_internal.h (in src)
 */
typedef struct layer layer;

/// Node for `layers_cache` singly linked list
typedef struct layers_cache_node {
    tensor* t;
    struct layers_cache_node* next;
} layers_cache_node;

/**
 * @brief Layers cache
 *
 * This is just a stack of `tensor`s used in layer feedforward and backprop functions. <br>
 * Layers use the cache if they need to transfer data from the feedforward to the backprop. <br>
 * This is necessary because of the multithreading.
 */
typedef struct layers_cache {
    /**
     * @brief Arena used for the cache
     * 
     * If a layer is pushing tensors onto the cache
     * the tensor should created with this arena
     */
    mg_arena* arena;

    /// First node of SLL
    layers_cache_node* first;
    /// Last node of SLL
    layers_cache_node* last;
} layers_cache;

/// Reshape layer backend
typedef struct {
    tensor_shape prev_shape;
} layer_reshape_backend;

/// Dense layer backend
typedef struct {
    tensor* weight;
    tensor* bias;

    // Training mode
    param_change weight_change;
    param_change bias_change;
} layer_dense_backend;

/// Activation layer backend
typedef struct {
    layer_activation_type type;
} layer_activation_backend;

/// Dropout layer backend
typedef struct {
    f32 keep_rate;
} layer_dropout_backend;

/// Flatten layer backend
typedef struct {
    tensor_shape prev_shape;
} layer_flatten_backend;

/// Pooling layer backend
typedef struct {
    tensor_shape input_shape;

    tensor_shape pool_size;
    layer_pooling_type type;
} layer_pooling_2d_backend;

/// 2D convolutional layer backend
typedef struct {
    u32 kernel_size;

    // Shape is (kernel_size * kernel_size, in_filters, out_filters)
    tensor* kernels;
    // Shape is out_shape
    tensor* biases; 

    u32 stride;
    u32 padding;

    tensor_shape input_shape;

    // Training mode
    param_change kernels_change;
    param_change biases_change;
} layer_conv_2d_backend;

/// Layer normalization backend
typedef struct {
    /// For numerical stability
    f32 epsilon;
} layer_norm_backend;

/// Layer structure. You usually do not have to worry about the internals of these
typedef struct layer {
    /// Initialized in layer_create
    layer_type type;
    b32 training_mode;

    /// Should be set by layer in create function
    tensor_shape shape;

    union {
        layer_reshape_backend reshape_backend;
        layer_dense_backend dense_backend;
        layer_activation_backend activation_backend;
        layer_dropout_backend dropout_backend;
        layer_flatten_backend flatten_backend;
        layer_pooling_2d_backend pooling_2d_backend;
        layer_conv_2d_backend conv_2d_backend;
        layer_norm_backend norm_backend;
    };
} layer;

/**
 * @brief Gets the name of a layer from the type
 * 
 * @return `string8` with the layer name. Do not modify the string data 
 */
string8 layer_get_name(layer_type type);
/**
 * @brief Gets hte layer type from the `name`
 *
 * @return `layer_type` correlated with `name`; LAYER_NULL if `name` is null or invalid
 */
layer_type layer_from_name(string8 name);

/**
 * @brief Creates a layer from a `layer_desc`
 *
 * @param arena Memory arena to create the layer in
 * @param desc Pointer to layer desc
 * @param prev_shape Shape of previous layer. This is required for many layers to work properly
 */
layer* layer_create(mg_arena* arena, const layer_desc* desc, tensor_shape prev_shape);
/**
 * @brief Feedforwards layer
 *
 * @param l Layer to be used
 * @param in_out Input to layer and where the output gets stored
 * @param cache Layer cache only used for training. Can be NULL
 */
void layer_feedforward(layer* l, tensor* in_out, layers_cache* cache); 
/**
 * @brief Backpropagation of layer
 *
 * Layer should be in training mode, and the cache is required
 *
 * @param l Layer to be used
 * @param delta Running gradient of backpropagation.
 *  The backprop function will update any layer params and the delta
 * @param cache Layer cache
 */
void layer_backprop(layer* l, tensor* delta, layers_cache* cache);
/**
 * @brief Applies any changes accumulated in backprop to layer
 *
 * @param l Layer to be used
 * @param optim Optimizer to be uzed
 */
void layer_apply_changes(layer* l, const optimizer* optim);
/**
 * @brief Deletes the layer
 *
 * This is annoying, but it is required for some multithreading stuff.
 *
 * @param l Layer to delete
 */
void layer_delete(layer* l);
/** 
 * @brief Saves any trainable params of the layer
 *
 * This does not include anything that would be in a desc.
 *
 * @param arena Arena for nodes in the `list`
 * @param l Layer to save
 * @param list List to save tensors to
 * @param index Index of the layer in neural network. To make names in the list unique
 */
void layer_save(mg_arena* arena, layer* l, tensor_list* list, u32 index);
/**
 * @brief Loads trainable params of the layer
 *
 * @param l The layer to load. The layer should be initialized with a desc
 * @param list List with the loaded tensors
 * @param index Index of the layer in the neural network
 */
void layer_load(layer* l, const tensor_list* list, u32 index);

/**
 * @brief Retrives the default desc of the layer type
 *
 * @return A copy of the default layer desc. It is okay to modify the return value.
 */
layer_desc layer_desc_default(layer_type type);
/**
 * @brief Applies defaults to parameters in the desc
 * 
 * @param desc A layer desc with the type set
 *
 * @return A new layer desc with default values for unset members of `desc`
 */
layer_desc layer_desc_apply_default(const layer_desc* desc);

/**
 * @brief Saves the desc to the `string8_list`
 *
 * Example format: `layer_type: field = value;`
 *
 * @param arena Arena for strings and nodes on the list
 * @param list Output string list for saving
 * @param desc Desc to save
 */
void layer_desc_save(mg_arena* arena, string8_list* list, const layer_desc* desc);
/**
 * @brief Loads the layer desc from the `string8`
 * 
 * @param out Output layer desc
 * @param str A valid layer desc str
 *
 * @return true if loading was successful
 */
b32 layer_desc_load(layer_desc* out, string8 str);

/**
 * @brief Initializes `param` based on the init type
 *
 * @param param The tensor to init
 * @param input_type Type of initialization
 * @param in_size Size of input to param/layer (e.g. `(u64)input->shape.width * input->shape.height * input->shape.depth`)
 * @param out_size Size of output of param/layer
 */
void param_init(tensor* param, param_init_type input_type, u64 in_size, u64 out_size);

/**
 * @brief Pushes the `tensor` onto the `layers_cache`
 */
void layers_cache_push(layers_cache* cache, tensor* t);
/** 
 * @brief Pops a `tensor` off of the `layers_cache` and returns it
 */
tensor* layers_cache_pop(layers_cache* cache);

#endif // LAYERS_H