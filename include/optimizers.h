//
// Created by Vishal Jha on 16/01/26.
//

/**
 * @file optimizers.h
 * @brief Parameter optimizers for the neural networks
 */

#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "base_defs.h"
#include "os.h"
#include "tensorNew.h"

/// Type of optimizers
typedef enum {
    /// Does nothing
    OPTIMIZER_NULL = 0,

    /// Stochastic Gradient Descent
    OPTIMIZER_SGD,

    /// Root Mean Square Propagation
    OPTIMIZER_RMS_PROP,

    /// Adaptive Moment Estimation
    OPTIMIZER_ADAM,

    /// Number of optimizers
    OPTIMIZER_COUNT
} optimizer_type;

/// Stochastic Gradient Descent Parameters
typedef struct {
    /**
     * @brief Exponentially moving average param
     * 
     * Typically `0.9` <br>
     * `0.0f <= momentum <= 1.0f` <br>
     * `V = momentum * V_prev + (1 - momentum) * dW`
     */ 
    f32 momentum;
} optimizer_sgd;

/// Root Mean Squared Propagation Parameters
typedef struct {
    /** 
     * @brief Discounting factor for old gradients
     *
     * Typically `0.999` <br>
     * `0.0f <= beta <= 1.0f` <br>
     * `S = beta * S_prev + (1 - beta) * (dW)^2` 
     */
    f32 beta;
    
    /**
     * @brief For numerical stability
     *
     * `W = W - learning_rate * (dW / sqrt(S + epsilon))`
     */
    f32 epsilon;
} optimizer_rms_prop;

/// Adaptive Moment Estimation Parameters
typedef struct {
    /**
     * @brief Exponentially moving average param
     *
     * See `optimizer_sgd` `momentum``
     */ 
    f32 beta1;

    /**
     * @brief Discounting factor for old gradients
     *
     * See `optimizer_rms_prop` `beta`
     */
    f32 beta2;

    /**
     * @brief For numerical stability
     *
     * See `optimizer_rms_prop` `epsilon`
     */
    f32 epsilon;
} optimizer_adam;

/// Full optimizer params
typedef struct {
    /**
     * @brief Scaling factor for changes
     * 
     * `W = W - learning_rate * (changes)`
     */
    f32 learning_rate;

    /// Type of optimizer. Used for accessing the union
    optimizer_type type;

    union {
        /// SGD params
        optimizer_sgd sgd;
        /// RMS Prop params
        optimizer_rms_prop rms_prop;
        /// Adam params
        optimizer_adam adam;
    };

    /**
     * @brief Batch size during learning
     *
     * Does not need to be set in network_train_desc (will be set by neural network)
     */
    u32 _batch_size;
} optimizer;

/**
 * @brief Storage for changes in trainable parameters
 *
 * Do not modify any members of a `param_change` directly. <br>
 * If a layer has trainable params, the changes should
 * be accumulated and applied with a `param_change`
 */
typedef struct {
    /// Mutex for changing the other params
    mutex* _mutex;

    /**
     * @brief Change in param
     * 
     * Should not be updated directly
     */
    tensor* _change;

    /// State for SGD and Adam
    tensor* _V;

    /// State for RMS Prop ans Adam
    tensor* _S;
} param_change;

/**
 * @brief Initializes a `param_change` in `out`
 *
 * This does not return a param_change because layers should already have a `param_change` member. <br>
 * See `layers_dense.c` for an example
 *
 * @param arena Arena for param_change
 * @param out Output of creation
 * @param shape Shape of param
 */
void param_change_create(mg_arena* arena, param_change* out, tensor_shape shape);
/**
 * @brief Adds `addend` to `param_change`
 *
 * Layers must use this function for thread safety
 */
void param_change_add(param_change* param_change, tensor* addend);
/**
 * @brief Applies any changes in `param_change` to `param`
 *
 * @param optim Optimizer to use for updating
 * @param param Parameter to update
 * @param param_change Param change for `param`
 */
void param_change_apply(const optimizer* optim, tensor* param, param_change* param_change);
/**
 * @brief Deletes `param_change`
 *
 * This is necessary because of the mutex
 */
void param_change_delete(param_change* param_change);

#endif // OPTIMIZERS_H