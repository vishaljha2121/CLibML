//
// Created by Vishal Jha on 16/01/26.
//
#ifndef NETWORK_H
#define NETWORK_H

#include "base_defs.h"
#include "str.h"
#include "../src/mg/mg_arena.h"
#include "tensorNew.h"

#include "layers.h"
#include "cost.h"
#include "optimizers.h"

/**
 * @brief Sequential neural network
 */
typedef struct {
    /// Whether or not training mode is enabled. Set during creation functions
    b32 training_mode;

    /// Number of layers
    u32 num_layers;
    /// Array of layers
    layer** layers;

    /**
     * @brief List of layer descs
     * 
     * Used for neural network saving
     */
    layer_desc* layer_descs;

    /**
     * @brief Used for forward and backward passes
     * Allows for single allocation of input/output variable
     */
    u64 max_layer_size;
} network;

/// Information about random transformations in the network training inputs
typedef struct {
    /// Minimum random translation, inclusize. Applied on both axes
    f32 min_translation;
    /// Maximum random translation, exclusize. Applied on both axes
    f32 max_translation;

    /// Minimum random scale, inclusive. Applied on both axes
    f32 min_scale;
    /// Maximum random scale, exclusive. Applied on both axes
    f32 max_scale;

    /// Minimum random angle in radians, inclusive
    f32 min_angle;
    /// Maximum random angle in radians, exclusive
    f32 max_angle;
} network_transforms;

/// Info for epoch callback
typedef struct {
    /// Epoch number. Starts at 0
    u32 epoch;

    /// Accuracy of test, if accuracy test is enabled in training
    f32 test_accuracy;
} network_epoch_info;

/// Callback function for training
typedef void(network_epoch_callback)(const network_epoch_info*);

/**
 * @brief Neural network training description
 *
 * For `network_train`
 */
typedef struct {
    /// Number of epochs to train
    u32 epochs;
    /// Size of training batch
    u32 batch_size;

    /// Number of threads to train on
    u32 num_threads;

    /// Cost function to use
    cost_type cost;
    /**
     * @brief Optimizer to use
     *
     * You do not have to set `batch_size` in the optimizer
     */
    optimizer optim;

    /// Whether or not to randomly transform the training inputs
    b32 random_transforms;
    /// Random transforms to be applied to training inputs
    network_transforms transforms;


    /// Callback function called after each epoch. Can be NULL 
    network_epoch_callback* epoch_callback;

    /**
     * @brief Epoch interval to save network
     *
     * If `save_interval` == 0, then the network does not save. <br>
     * Saves when `(epoch + 1) % save_interval == 0`
     */
    u32 save_interval;
    /**
     * @brief Output path of save interval
     *
     * Output file is `{save_path}{epoch}.tsn`
     */
    string8 save_path;

    /**
     * @brief Training inputs to neural network.
     *
     * One training input is a 2D slice of the `train_inputs` tensor.
     * If you have 3D inputs, reduce them to 2D for `train_inputs`
     * then resize them in the input layer of the neural network
     */
    tensor* train_inputs;
    /**
     * @brief Training outputs of neural network.
     *
     * 2D slices are taken for each output.
     * Depth must be the same as the depth of `train_inputs`
     */
    tensor* train_outputs;

    /// Whether or not to enable an accuracy test after each epoch
    b32 accuracy_test;
    /**
     * @brief Inputs for testing
     *
     * Same shape requirements as `train_inputs` apply
     */
    tensor* test_inputs;
    /**
     * @brief Outputs for testing
     *
     * Same shape requirements as `train_outputs` apply
     */
    tensor* test_outputs;
} network_train_desc;

// This training_mode overrides the one in the desc
/**
 * @brief Creates a neural network
 *
 * @param arena Arena to create network on
 * @param num_layers Number of layers and size of the `layer_descs` array
 * @param layer_descs List of layer descriptions
 * @param training_mode Whether or not to initialize the network in training mode.
 *  This overrides the training mode in the layer descs
 *
 * @return Pointer to network on success, NULL on failure
 */
network* network_create(mg_arena* arena, u32 num_layers, const layer_desc* layer_descs, b32 training_mode);
/**
 * @brief Creates a network from a layout file (.tsl)
 *
 * Layout files can be created by hand or by `network_save_layout`
 *
 * @param arena Arena to create network on
 * @param file_name File to load
 * @param training_mode Whether or not to initalize the network in training mode
 *
 * @return Pointer to network on success, NULL on failure
 */
network* network_load_layout(mg_arena* arena, string8 file_name, b32 training_mode);
/**
 * @brief Creates a network from a network file (.tsn)
 *
 * Network files are created by `network_save`,
 * and they include the parameters of the neural network.
 * Used to load a network that has already been trained.
 *
 * @param arena Arena to create network on
 * @param file_name File to load
 * @param training_mode Whether or not to initalize the network in training mode
 *
 * @return Pointer to network on success, NULL on failure
 */
network* network_load(mg_arena* arena, string8 file_name, b32 training_mode);

void network_load_existing(network* nn, string8 file_name);

/**
 * @brief Deletes the neural network
 *
 * This is annoying, but required because of some threading stuff
 */
void network_delete(network* nn);

/**
 * @brief Feeds `input` through the network, and puts the result in `out`
 *
 * @param nn Network to use
 * @param out Output of feedforward. Must be big enough
 * @param input Input to network
 */
void network_feedforward(const network* nn, tensor* out, const tensor* input);

/**
 * @brief Trains the neural network based on the training description
 *
 * See `network_train_desc` for details
 *
 * @param nn Network to train
 * @param desc Training description
 */
void network_train(network* nn, const network_train_desc* desc);

/**
 * @brief Prints a summary of the network to stdout
 *
 * Shows the layer types and shapes
 */
void network_summary(const network* nn);

/**
 * @brief Saves the layout of the network into a .tsl file
 *
 * Saves any information stored in the layer descriptions
 *
 * @param nn Network to save layout
 * @param file_name Output of save layout. This should include the file extension
 */
void network_save_layout(const network* nn, string8 file_name);

/**
 * @brief Saves the network into a .tsn file
 *
 * Saves layout and parameter information.
 * Usually used during or after training the network
 *
 * @param nn Network to save
 * @param file_name File to save to. This shoudl include the file extension
 */
void network_save(const network* nn, string8 file_name);

#endif // NETWORK_H