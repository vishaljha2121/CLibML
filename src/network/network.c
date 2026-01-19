//
// Created by Vishal Jha on 16/01/26.
//
#include "../../include/network.h"
#include "../layers/layers_internal.h"
#include "../../include/err.h"
#include "../../include/img.h"
#include "../random_generators/prng.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

u32 _network_max_layer_size(const network* nn) {
    u64 max_layer_size = 0;
    for (u32 i = 0; i < nn->num_layers; i++) {
        tensor_shape s = nn->layers[i]->shape;

        u64 size = (u64)s.width * s.height * s.depth;

        if (size > max_layer_size) {
            max_layer_size = size;
        }
    }

    return max_layer_size;
}

// Checks that each layer outputs the correct shape
// Called after input layer check and max_layer_size
b32 _network_shape_checks(const network* nn) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Performing a mock feedforward and backprop to check shapes
    tensor* in_out = tensor_create_alloc(scratch.arena, nn->layers[0]->shape, nn->max_layer_size);
    layers_cache cache = { .arena = scratch.arena };

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_feedforward(nn->layers[i], in_out, &cache);

        if (!tensor_shape_eq(in_out->shape, nn->layers[i]->shape)) {
            goto fail;
        }
    }

    if (nn->training_mode) {
        // Renaming for clarity
        tensor* delta = in_out;

        for (i64 i = nn->num_layers - 1; i >= 0; i--) {
            layer_backprop(nn->layers[i], delta, &cache);

            if (i != 0 && !tensor_shape_eq(delta->shape, nn->layers[i-1]->shape)) {
                goto fail;
            }
        }
    }

    mga_scratch_release(scratch);
    return true;

fail:
    mga_scratch_release(scratch);
    return false;
}

network* network_create(mg_arena* arena, u32 num_layers, const layer_desc* layer_descs, b32 training_mode) {
    if (layer_descs == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot create network with NULL layer_descs");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);
    network* nn = MGA_PUSH_ZERO_STRUCT(arena, network);

    nn->training_mode = training_mode;
    nn->num_layers = num_layers;

    nn->layer_descs = MGA_PUSH_ZERO_ARRAY(arena, layer_desc, nn->num_layers);
    nn->layers = MGA_PUSH_ZERO_ARRAY(arena, layer*, nn->num_layers);

    tensor_shape prev_shape = { 0 };
    for (u32 i = 0; i < nn->num_layers; i++) {
        nn->layer_descs[i] = layer_desc_apply_default(&layer_descs[i]);
        nn->layer_descs[i].training_mode = training_mode;

        nn->layers[i] = layer_create(arena, &nn->layer_descs[i], prev_shape);

        if (nn->layers[i] == NULL) {
            ERR(ERR_CREATE, "Cannot create network: failed to create layer");

            goto error;
        }

        prev_shape = nn->layers[i]->shape;
    }

    if (nn->layers[0]->type != LAYER_INPUT) {
        ERR(ERR_INVALID_INPUT, "First layer of network must be input");
        goto error;
    }

    nn->max_layer_size = _network_max_layer_size(nn); 

    if (!_network_shape_checks(nn)) {
        ERR(ERR_INVALID_INPUT, "Cannot create network: layer shapes do not align");
        goto error;
    }

    return nn;

error:
    mga_temp_end(maybe_temp);
    return NULL;
}

// Inits layers from stripped tsl string
// See network_save_layout for more detail
static b32 _network_load_layout_impl(mg_arena* arena, network* nn, string8 file, b32 training_mode) {
    mga_temp scratch = mga_scratch_get(&arena, 1);

    // Each string in list is a layer_desc save str
    string8_list desc_str_list = { 0 };

    u64 desc_str_start = 0;
    u64 last_semi = 0;
    b32 first_colon = true;
    for (u64 i = 0; i < file.size; i++) {
        u8 c = file.str[i];

        if (c == ';') {
            last_semi = i;

            continue;
        }

        // Colon triggers start of new desc
        // So the old one gets pushed onto the list
        if (c == ':') {
            // If it is the first colon, the string should not be saved 
            if (first_colon) {
                first_colon = false;

                continue;
            }

            string8 desc_str = str8_substr(file, desc_str_start, last_semi + 1);

            str8_list_push(scratch.arena, &desc_str_list, desc_str);

            desc_str_start = last_semi + 1;

            // This makes it so that layers without parameters still work correctly
            // (Layers without params would have no semi colons)
            last_semi = i;
        }
    }
    string8 last_str = str8_substr(file, desc_str_start, file.size);
    str8_list_push(scratch.arena, &desc_str_list, last_str);

    nn->num_layers = desc_str_list.node_count;

    nn->layer_descs = MGA_PUSH_ZERO_ARRAY(arena, layer_desc, nn->num_layers);
    nn->layers = MGA_PUSH_ZERO_ARRAY(arena, layer*, nn->num_layers);

    string8_node* n = desc_str_list.first;
    tensor_shape prev_shape = { 0 };
    for (u32 i = 0; i < nn->num_layers; i++, n = n->next) {
        if (!layer_desc_load(&nn->layer_descs[i], n->str)) {
            goto error;
        }

        nn->layer_descs[i] = layer_desc_apply_default(&nn->layer_descs[i]);
        nn->layer_descs[i].training_mode = training_mode;

        nn->layers[i] = layer_create(arena, &nn->layer_descs[i], prev_shape);

        if (nn->layers[i] == NULL) {
            goto error;
        }

        prev_shape = nn->layers[i]->shape;
    }

    if (nn->layers[0]->type != LAYER_INPUT) {
        ERR(ERR_INVALID_INPUT, "First layer of network must be input");

        goto error;
    }

    nn->max_layer_size = _network_max_layer_size(nn); 

    mga_scratch_release(scratch);

    if (!_network_shape_checks(nn)) {
        ERR(ERR_INVALID_INPUT, "Cannot create network: layer shapes do not align");
        goto error;
    }
    
    return true;

error:
    mga_scratch_release(scratch);
    return false;
}

// Creates network from layout file (*.tsl)
network* network_load_layout(mg_arena* arena, string8 file_name, b32 training_mode) {
    mga_temp maybe_temp = mga_temp_begin(arena);
    network* nn = MGA_PUSH_ZERO_STRUCT(arena, network);

    nn->training_mode = training_mode;

    mga_temp scratch = mga_scratch_get(&arena, 1);

    string8 raw_file = file_read(scratch.arena, file_name);
    string8 file = str8_remove_space(scratch.arena, raw_file);

    if (!_network_load_layout_impl(arena, nn, file, training_mode)) {
        mga_temp_end(maybe_temp);
        mga_scratch_release(scratch);

        return NULL;
    }

    mga_scratch_release(scratch);

    return nn;
}

// This is also used in network_save
static const string8 _tsn_header = {
    .size = 10,
    .str = (u8*)"network"
};

// Creates network from network file (*.tsn)
network* network_load(mg_arena* arena, string8 file_name, b32 training_mode) {
    mga_temp maybe_temp = mga_temp_begin(arena);
    network* nn = MGA_PUSH_ZERO_STRUCT(arena, network);

    nn->training_mode = training_mode;

    mga_temp scratch = mga_scratch_get(&arena, 1);

    string8 file = file_read(scratch.arena, file_name);

    if (!str8_equals(_tsn_header, str8_substr(file, 0, _tsn_header.size))) {
        ERR(ERR_PARSE, "Cannot load network: not tsn file");

        goto error;
    }

    file = str8_substr(file, _tsn_header.size, file.size);

    u64 tst_index = 0;
    if (!str8_index_of(file, tensor_get_tst_header(), &tst_index)) {
        ERR(ERR_PARSE, "Cannot load network: invalid tsn file");

        goto error;
    }

    string8 layout_str = str8_substr(file, 0, tst_index);
    string8 tensors_str = str8_substr(file, tst_index, file.size);

    if (!_network_load_layout_impl(arena, nn, layout_str, training_mode)) {
        goto error;
    }

    tensor_list params = tensor_list_from_str(scratch.arena, tensors_str);

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_load(nn->layers[i], &params, i);
    }

    mga_scratch_release(scratch);
    return nn;

error:
    mga_temp_end(maybe_temp);
    mga_scratch_release(scratch);

    return NULL;
}

void network_delete(network* nn) {
    if (nn == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot delete NULL network");
        return;
    }

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_delete(nn->layers[i]);
    }
}

void network_feedforward(const network* nn, tensor* out, const tensor* input) {
    if (nn == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot feedforward NULL network");
        return;
    }
    if (out == NULL || input == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot feedforward with NULL input and/or output");
        return;
    }

    u64 input_size = (u64)input->shape.width * input->shape.height * input->shape.depth;
    tensor_shape nn_shape = nn->layers[0]->shape;
    u64 nn_input_size = (u64)nn_shape.width * nn_shape.height * nn_shape.depth;

    if (input_size != nn_input_size) {
        ERR(ERR_INVALID_INPUT, "Input must be as big as the network input layer");
        return;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    tensor* in_out = tensor_create_alloc(scratch.arena, (tensor_shape){ 1, 1, 1 }, nn->max_layer_size);
    tensor_copy_ip(in_out, input);

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_feedforward(nn->layers[i], in_out, NULL);
    }

    tensor_copy_ip(out, in_out);

    mga_scratch_release(scratch);
}

u32 _num_digits (u32 n) {
    if (n < 10) return 1;
    if (n < 100) return 2;
    if (n < 1000) return 3;
    if (n < 10000) return 4;
    if (n < 100000) return 5;
    if (n < 1000000) return 6;
    if (n < 10000000) return 7;
    if (n < 100000000) return 8;
    if (n < 1000000000) return 9;
    return 10;
}

typedef struct {
    network* nn;
    b32 apply_transforms;
    const network_transforms* transforms;

    tensor input_view;
    tensor output_view;

    cost_type cost;
} _network_backprop_args;

void _network_backprop_thread(void* args) {
    _network_backprop_args* bargs = (_network_backprop_args*)args;

    network* nn = bargs->nn;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    layers_cache cache = { .arena = scratch.arena };

    tensor* in_out = tensor_create_alloc(scratch.arena, (tensor_shape){ 1, 1, 1 }, nn->max_layer_size);
    tensor_copy_ip(in_out, &bargs->input_view);

    // Setting input shape to input layer shape
    // This should all be okay because of the checks when creating a network
    in_out->shape = nn->layers[0]->shape;

    if (bargs->apply_transforms) {
        const network_transforms* t = bargs->transforms;

        f32 x_off = t->min_translation + prng_rand_f32() * (t->max_translation - t->min_translation);
        f32 y_off = t->min_translation + prng_rand_f32() * (t->max_translation - t->min_translation);

        f32 x_scale = t->min_scale + prng_rand_f32() * (t->max_scale - t->min_scale);
        f32 y_scale = t->min_scale + prng_rand_f32() * (t->max_scale - t->min_scale);

        f32 angle = t->min_angle + prng_rand_f32() * (t->max_angle - t->min_angle);
        f32 sin_a = sinf(angle);
        f32 cos_a = cosf(angle);

        img_mat3 mat = {
            .m = {
                x_scale * cos_a, y_scale * -sin_a, x_off,
                x_scale * sin_a, y_scale *  cos_a, y_off,
                0, 0, 1
            }
        };

        img_transform_ip(in_out, in_out, SAMPLE_BILINEAR, &mat);
    }

    tensor* output = tensor_copy(scratch.arena, &bargs->output_view, false);

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_feedforward(nn->layers[i], in_out, &cache);
    }

    // Renaming for clarity
    tensor* delta = in_out;
    cost_grad(bargs->cost, delta, output);

    for (i64 i = nn->num_layers - 1; i >= 0; i--) {
        layer_backprop(nn->layers[i], delta, &cache);
    }

    mga_scratch_release(scratch);
}

typedef struct {
    u32* num_correct;
    mutex* num_correct_mutex;

    network* nn;

    tensor input_view;
    tensor_index output_argmax;
} _network_test_args;
void _network_test_thread(void* args) {
    _network_test_args* targs = (_network_test_args*)args;

    network* nn = targs->nn;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    tensor* in_out = tensor_create_alloc(scratch.arena, (tensor_shape){ 1, 1, 1 }, nn->max_layer_size);
    tensor_copy_ip(in_out, &targs->input_view);

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_feedforward(nn->layers[i], in_out, NULL);
    }

    if (tensor_index_eq(tensor_argmax(in_out), targs->output_argmax)) {
        mutex_lock(targs->num_correct_mutex);

        *targs->num_correct += 1;

        mutex_unlock(targs->num_correct_mutex);
    }    

    mga_scratch_release(scratch);
}

#define _BAR_SIZE 20
void network_train(network* nn, const network_train_desc* desc) {
    if (nn == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot train NULL network");
        return;
    }
    if (!nn->training_mode) {
        ERR(ERR_INVALID_INPUT, "Cannot train network that is not in training mode");
        return;
    }

    // Size checks
    {
        tensor_shape nn_shape = nn->layers[0]->shape;
        u64 nn_input_size = (u64)nn_shape.width * nn_shape.height * nn_shape.depth;
        nn_shape = nn->layers[nn->num_layers - 1]->shape;
        u64 nn_out_size = (u64)nn_shape.width * nn_shape.height * nn_shape.depth;

        u64 input_size = (u64)desc->train_inputs->shape.width * desc->train_inputs->shape.height;
        if (input_size != nn_input_size) {
            ERR(ERR_INVALID_INPUT, "Training inputs must be the same size as the network input layer");
            return;
        }
        u64 out_size = (u64)desc->train_outputs->shape.width * desc->train_outputs->shape.height;
        if (out_size != nn_out_size) {
            ERR(ERR_INVALID_INPUT, "Training outpus must be the same size as the network output layer");
            return;
        }

        if (desc->accuracy_test) {
            input_size = (u64)desc->test_inputs->shape.width * desc->test_inputs->shape.height;
            if (input_size != nn_input_size) {
                ERR(ERR_INVALID_INPUT, "Testing inputs must be the same size as the network input layer");
                return;
            }
            out_size = (u64)desc->test_outputs->shape.width * desc->test_outputs->shape.height;
            if (out_size != nn_out_size) {
                ERR(ERR_INVALID_INPUT, "Testing outpus must be the same size as the network output layer");
                return;
            }
        }
    }

    optimizer optim = desc->optim;
    optim._batch_size = desc->batch_size;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // +1 is just for insurance
    thread_pool* tpool = thread_pool_create(scratch.arena, MAX(1, desc->num_threads), desc->batch_size + 1);

    _network_backprop_args* backprop_args = MGA_PUSH_ZERO_ARRAY(scratch.arena, _network_backprop_args, desc->batch_size);

    // Accuracy testing stuff
    _network_test_args* test_args = NULL;
    u32 num_correct = 0;
    mutex* num_correct_mutex = NULL;
    if (desc->accuracy_test) {
        test_args = MGA_PUSH_ZERO_ARRAY(scratch.arena, _network_test_args, desc->batch_size);
        num_correct_mutex = mutex_create(scratch.arena);
    }

    u8 bar_str_data[_BAR_SIZE + 1] = { 0 };
    memset(bar_str_data, ' ', _BAR_SIZE);

    u8 batch_str_data[11] = { 0 };

    // This will add one if there is a remainder
    // Allows for batch sizes that are not perfectly divisible
    div_t num_batches_div = div(desc->train_inputs->shape.depth, desc->batch_size);
    u32 num_batches = num_batches_div.quot + (num_batches_div.rem != 0);
    u32 last_batch_size = desc->train_inputs->shape.depth - (desc->batch_size * (num_batches - 1));

    u32 num_batches_digits = _num_digits(num_batches);

    // Same calculations for test batches
    div_t num_test_batches_div = div(desc->test_inputs->shape.depth, desc->batch_size);
    u32 num_test_batches = num_test_batches_div.quot + (num_test_batches_div.rem != 0);
    u32 last_test_batch_size = desc->test_inputs->shape.depth - (desc->batch_size * (num_test_batches - 1));

    time_init();

    for (u32 epoch = 0; epoch < desc->epochs; epoch++) {
        printf("Epoch: %u / %u\n", epoch + 1, desc->epochs);

        u64 batch_start = now_usec();

        for (u32 batch = 0; batch < num_batches; batch++) {
            // Progress in stdout
            {
                // This is so the batch number always takes up the same amount of space
                u32 batch_digits = _num_digits(batch + 1);
                memset(batch_str_data, ' ', 9);
                u32 offset = num_batches_digits - batch_digits;
                snprintf((char*)(batch_str_data + offset), 11 - offset, "%u", batch + 1);
                printf("%.*s / %u  ", (int)num_batches_digits, batch_str_data, num_batches);

                f32 bar_length = (f32)_BAR_SIZE * ((f32)(batch + 1) / num_batches);
                u32 bar_chars = ceilf(bar_length);
                memset(bar_str_data, '=', bar_chars);
                if (batch + 1 != num_batches) {
                    bar_str_data[bar_chars - 1] = '>';
                }

                printf("[%s]", bar_str_data);
                
                if (batch != 0) {
                    u64 cur_time = now_usec();
                    f32 elapsed = (cur_time - batch_start) / 1e6f;
                    f32 per_batch = elapsed / batch;
                    u32 etm = (u32)(per_batch * (num_batches - batch));

                    printf(" ETM -- %02u:%02u:%02u", etm / (3600), (etm % 3600) / 60, etm % 60);
                }

                printf("\r");

                fflush(stdout);
            }

            mga_temp batch_temp = mga_temp_begin(scratch.arena);

            // Training batch
            u32 batch_size = (batch == num_batches - 1) ? last_batch_size : desc->batch_size;
            for (u32 i = 0; i < batch_size; i++) {
                u64 index = (u64)i + (u64)batch * desc->batch_size;

                tensor input_view = { 0 };
                tensor output_view = { 0 };
                tensor_2d_view(&input_view, desc->train_inputs, index);
                tensor_2d_view(&output_view, desc->train_outputs, index);

                backprop_args[i] = (_network_backprop_args){ 
                    .nn = nn,
                    .apply_transforms = desc->random_transforms,
                    .transforms = &desc->transforms,
                    .input_view = input_view,
                    .output_view = output_view,
                    .cost = desc->cost,
                };

                thread_pool_add_task(
                    tpool,
                    (thread_task){
                        .func = _network_backprop_thread,
                        .arg = &backprop_args[i]
                    }
                );
            }

            thread_pool_wait(tpool);

            for (u32 i = 0; i < nn->num_layers; i++) {
                layer_apply_changes(nn->layers[i], &optim);
            }

            mga_temp_end(batch_temp);
        }

        printf("\n");
        memset(bar_str_data, ' ', _BAR_SIZE);

        if (desc->save_interval != 0 && ((epoch + 1) % desc->save_interval) == 0) {
            mga_temp save_temp = mga_temp_begin(scratch.arena);

            string8 path = str8_pushf(save_temp.arena, "%.*s%.4u.tsn", (int)desc->save_path.size, desc->save_path.str, epoch + 1);

            network_save(nn, path);

            mga_temp_end(save_temp);
        }

        f32 accuracy = 0.0f;
        if (desc->accuracy_test) {
            num_correct = 0;

            string8 load_anim = STR8("-\\|/");
            u64 anim_start_time = now_usec();
            u32 anim_frame = 0;

            // Accuracy test is also done in batches for multithreading
            for (u32 batch = 0; batch < num_test_batches; batch++) {
                u64 cur_time = now_usec();
                if (cur_time - anim_start_time > 100000) {
                    anim_start_time = cur_time;
                    anim_frame++;

                    printf("Test Accuracy: %c\r", load_anim.str[anim_frame % load_anim.size]);
                    fflush(stdout);
                }

                mga_temp batch_temp = mga_temp_begin(scratch.arena);

                // Test batch
                u32 batch_size = batch == num_test_batches - 1 ? last_test_batch_size : desc->batch_size;
                for (u32 i = 0; i < batch_size; i++) {
                    u64 index = (u64)i + (u64)batch * desc->batch_size;

                    tensor input_view = { 0 };
                    tensor output_view = { 0 };
                    tensor_2d_view(&input_view, desc->test_inputs, index);
                    tensor_2d_view(&output_view, desc->test_outputs, index);

                    tensor_index output_argmax = tensor_argmax(&output_view);

                    test_args[i] = (_network_test_args){ 
                        .num_correct = &num_correct,
                        .num_correct_mutex = num_correct_mutex,

                        .nn = nn,
                        .input_view = input_view,
                        .output_argmax = output_argmax,
                    };

                    thread_pool_add_task(
                        tpool,
                        (thread_task){
                            .func = _network_test_thread,
                            .arg = &test_args[i]
                        }
                    );
                }

                thread_pool_wait(tpool);

                mga_temp_end(batch_temp);
            }

            accuracy = (f32)num_correct / desc->test_inputs->shape.depth;

            printf("Test Accuracy: %f\n", accuracy);
        }

        if (desc->epoch_callback) {
            network_epoch_info info = {
                .epoch = epoch,

                .test_accuracy = accuracy
            };

            desc->epoch_callback(&info);
        }
    }

    thread_pool_destroy(tpool);

    if (desc->accuracy_test) {
        mutex_destroy(num_correct_mutex);
    }

    mga_scratch_release(scratch);
}

/*
Sample Summary:

-------------------------
  network (5 layers)

type        shape
----        -----
input       (784, 1, 1)
dense       (64, 1, 1)
activation  (64, 1, 1)
dense       (10, 1, 1)
activation  (10, 1, 1)

-------------------------
*/
void network_summary(const network* nn) {
    if (nn == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot print summary of NULL network");
        return;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    string8 header = str8_pushf(scratch.arena, "network (%u layers)", nn->num_layers);

    // Storing strings in a list first to get good spacing in the console
    // +2 is for column name and "---" separator
    string8* types = MGA_PUSH_ZERO_ARRAY(scratch.arena, string8, nn->num_layers + 2);
    string8* shapes = MGA_PUSH_ZERO_ARRAY(scratch.arena, string8, nn->num_layers + 2);

    types[0] = STR8("type");
    types[1] = STR8("----");

    shapes[0] = STR8("shape");
    shapes[1] = STR8("-----");

    for (u32 i = 0; i < nn->num_layers; i++) {
        types[i + 2] = layer_get_name(nn->layers[i]->type);

        tensor_shape s = nn->layers[i]->shape;
        string8 shape_str = str8_pushf(scratch.arena, "(%u %u %u)", s.width, s.height, s.depth);

        shapes[i + 2] = shape_str;
    }

    u64 max_type_width = types[0].size;
    u64 max_shape_width = shapes[0].size;

    for (u32 i = 0; i < nn->num_layers; i++) {
        if (types[i + 2].size > max_type_width) {
            max_type_width = types[i + 2].size;
        }

        if (shapes[i + 2].size > max_shape_width) {
            max_shape_width = shapes[i + 2].size;
        }
    }

    // Spacing added before, between, and after items
    u64 row_width = 1 + max_type_width + 2 + max_shape_width + 1;
    row_width = MAX(row_width, header.size + 2);

    // For even spacing of the header
    if ((row_width - header.size) % 2 != 0) {
        row_width += 1;
    }
    
    // For newline
    row_width++;

    // Borders + border padding + header + layers + titles
    u32 num_rows = 2 + 2 + 1 + nn->num_layers + 2;

    string8 out = {
        .size = row_width * num_rows,
        .str = MGA_PUSH_ARRAY(scratch.arena, u8, row_width * num_rows)
    };

    memset(out.str, ' ', out.size);
    for (u32 y = 0; y < num_rows; y++) {
        out.str[row_width - 1 + y * row_width] = '\n';
    }

    // Borders
    memset(out.str, '-', row_width - 1);
    memset(out.str + (num_rows - 1) * row_width, '-', row_width - 1);

    // Header
    u32 header_spacing = (row_width - 1 - header.size) / 2;
    memcpy(out.str + row_width + header_spacing, header.str, header.size);

    u32 shape_start_x = 1 + max_type_width + 2;
    for (u32 i = 0; i < nn->num_layers + 2; i++) {
        // Start index into row
        u64 start_i = (i + 3) * row_width;

        memcpy(out.str + start_i + 1, types[i].str, types[i].size);
        memcpy(out.str + start_i + shape_start_x, shapes[i].str, shapes[i].size);
    }

    printf("%.*s", (int)out.size, (char*)out.str);

    mga_scratch_release(scratch);
}

/*
File Format (*.tsl):

List of layer_desc saves
See layer_desc_save
*/
void network_save_layout(const network* nn, string8 file_name) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    string8_list save_list = { 0 };

    // For spacing between layer_descs
    string8 new_line = STR8("\n");

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_desc_save(scratch.arena, &save_list, &nn->layer_descs[i]);
        str8_list_push(scratch.arena, &save_list, new_line);
    }

    file_write(file_name, save_list);

    mga_scratch_release(scratch);
}

/*
File Format (*.tsn):

Header
network Layout (tsl)
tensor List of layer params
*/
void network_save(const network* nn, string8 file_name) {
    if (nn == NULL) {
        ERR(ERR_INVALID_INPUT, "Cannot save NULL network");
        return;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);
    string8 layout_str = { 0 };

    {
        mga_temp scratch2 = mga_scratch_get(&scratch.arena, 1);

        string8_list layout_list = { 0 };
        for (u32 i = 0; i < nn->num_layers; i++) {
            layer_desc_save(scratch.arena, &layout_list, &nn->layer_descs[i]);
        }

        string8 full_layout_str = str8_concat(scratch2.arena, layout_list);
        layout_str = str8_remove_space(scratch.arena, full_layout_str);

        mga_scratch_release(scratch2);
    }


    tensor_list param_list = { 0 };
    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_save(scratch.arena, nn->layers[i], &param_list, i);
    }

    string8 param_str = tensor_list_to_str(scratch.arena, &param_list);

    string8_list save_list = { 0 };
    str8_list_push(scratch.arena, &save_list, _tsn_header);
    str8_list_push(scratch.arena, &save_list, layout_str);
    str8_list_push(scratch.arena, &save_list, param_str);

    file_write(file_name, save_list);

    mga_scratch_release(scratch);
}