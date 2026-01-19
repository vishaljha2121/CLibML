//
// Created by Vishal Jha on 16/01/26.
//

/**
 * @file img.h
 * @brief Image operations
 *
 * In each function, an "image" is just a tensor.
 * Each function should work on all color channels (i.e. 2d slices) of each tensor.
 * Values out of bounds in each image are assumed to be zero
 */ 

#ifndef IMG_H
#define IMG_H

#include "base_defs.h"
#include "tensorNew.h"

/// Sampling methods for image transformations
typedef enum {
    /// Sample nearest pixel
    SAMPLE_NEAREST,
    /// Linear interpolate pixels
    SAMPLE_BILINEAR
} img_sample_type;

/// Row major 3d matrix
typedef struct {
    f32 m[9];
} img_mat3;

/**
 * @brief Transforms the input image according to the matrix. In place version
 *
 * Transformations take place about the image's center.
 * Values out of bounds in each image are assumed to be zero
 *
 * @param out Output of transform. The output will be the same size as the input.
 * @param input Input image
 * @param sample_type Sampling type
 * @param mat Transformation matrix
 * 
 * @return true if out is big enough, false otherwise
 */
b32 img_transform_ip(tensor* out, const tensor* input, img_sample_type sample_type, const img_mat3* mat);

/**
 * @brief Translates the input image. In place version
 *
 * See `img_transform_ip` for details
 *
 * @param x_off Translation in x-axis
 * @param y_off Translation in y-axis
 */
b32 img_translate_ip(tensor* out, const tensor* input, img_sample_type sample_type, f32 x_off, f32 y_off);

/**
 * @brief Scales the input image. In place version
 *
 * See `img_transform_ip` for details
 *
 * @param x_scale Scale on x-axis
 * @param y_scale Scale on y-axis
 */
b32 img_scale_ip(tensor* out, const tensor* input, img_sample_type sample_type, f32 x_scale, f32 y_scale);

/**
 * @brief Rotates the input image. In place version
 *
 * See `img_transform_ip` for details
 *
 * @param theta Angle to rotate by, in radians
 */
b32 img_rotate_ip(tensor* out, const tensor* input, img_sample_type sample_type, f32 theta);

/**
 * @brief Shears the input image. In place version
 *
 * See `img_transform_ip` for details
 *
 * @param x_shear Amount to shear on x axis
 * @param y_shear Amount to shear on y axis
 */
b32 img_shear_ip(tensor* out, const tensor* input, img_sample_type sample_type, f32 x_shear, f32 y_shear);

/**
 * @brief Transforms the input image according to the matrix
 *
 * Transformations take place about the image's center.
 * Values out of bounds in each image are assumed to be zero
 *
 * @param arena Arena to create transformed image on
 * @param input Input image
 * @param sample_type Sampling type
 * @param mat Transformation matrix
 * 
 * @return The transformed image if successful, NULL otherwise
 */
tensor* img_transform(mg_arena* arena, const tensor* input, img_sample_type sample_type, const img_mat3* mat);

/**
 * @brief Translates the input image
 *
 * See `img_transform` for details
 *
 * @param x_off Translation in x-axis
 * @param y_off Translation in y-axis
 */
tensor* img_translate(mg_arena* arena, const tensor* input, img_sample_type sample_type, f32 x_off, f32 y_off);

/**
 * @brief Scales the input image
 *
 * See `img_transform` for details
 *
 * @param x_scale Scale on x-axis
 * @param y_scale Scale on y-axis
 */
tensor* img_scale(mg_arena* arena, const tensor* input, img_sample_type sample_type, f32 x_scale, f32 y_scale);

/**
 * @brief Rotates the input image
 *
 * See `img_transform` for details
 *
 * @param theta Angle to rotate by, in radians
 */
tensor* img_rotate(mg_arena* arena, const tensor* input, img_sample_type sample_type, f32 theta);

/**
 * @brief Shears the input image
 *
 * See `img_transform` for details
 *
 * @param x_shear Amount to shear on x axis
 * @param y_shear Amount to shear on y axis
 */
tensor* img_shear(mg_arena* arena, const tensor* input, img_sample_type sample_type, f32 x_shear, f32 y_shear);

#endif // IMG_H