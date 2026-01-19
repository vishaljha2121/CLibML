//
// Created by Vishal Jha on 16/01/26.
//

/**
 * @file costs.h
 * @brief Costs of neural networks
 */

#ifndef COST_H
#define COST_H

#include "base_defs.h"
#include "tensorNew.h"

/**
 * @brief Cost types
 */
typedef enum {
    /**
     * @brief Does not perform any operation on the inputs
     */
    COST_NULL = 0,

    /**
     * @brief Computes the mean squared error
     * 
     * C(a, y) = 0.5(a - y)^2 <br>
     * C'(a, y) = a - y
     */
    COST_MEAN_SQUARED_ERROR,

    /**
     * @brief Computes the categorical cross entorpy error
     * 
     * C(a, y) = y * ln(a) <br>
     * C'(a, y) = -y / a
     */
    COST_CATEGORICAL_CROSS_ENTROPY,

    /// Number for costs
    COST_COUNT
} cost_type;

/**
 * @brief Computes the cost
 *
 * `in` and `desired_out` must be the same shape
 *
 * @param type Which cost function to use
 * @param in Input of cost function (typically the neural network output)
 * @param desired_out True value of input (typically from some training data)
 */
f32 cost_func(cost_type type, const tensor* in, const tensor* desired_out);
/**
 * @brief Computes the gradient of the cost function
 * 
 * `in_out` and `desired_out` must be the same shape
 *
 * @param type Which cost function to use
 * @param in_out Input to cost gradient and where the output will be stored
 * @param desired_out True value of input (typically from some training data)
 */
void cost_grad(cost_type type, tensor* in_out, const tensor* desired_out);

#endif // COST_H