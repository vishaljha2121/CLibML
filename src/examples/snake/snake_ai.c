#include "snake_ai.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../../../include/layers.h"

// Helper to copy weights from src network to dest network
static void _snake_update_target_net(network* dest, const network* src) {
    if (dest->num_layers != src->num_layers) return;
    
    for (u32 i = 0; i < dest->num_layers; i++) {
        layer* l_dst = dest->layers[i];
        layer* l_src = src->layers[i];
        
        if (l_dst->type != l_src->type) continue;
        
        // Use backend specific copying
        switch (l_dst->type) {
            case LAYER_DENSE:
                tensor_copy_ip(l_dst->dense_backend.weight, l_src->dense_backend.weight);
                tensor_copy_ip(l_dst->dense_backend.bias, l_src->dense_backend.bias);
                break;
            case LAYER_CONV_2D:
                tensor_copy_ip(l_dst->conv_2d_backend.kernels, l_src->conv_2d_backend.kernels);
                tensor_copy_ip(l_dst->conv_2d_backend.biases, l_src->conv_2d_backend.biases);
                break;
            default: break;
        }
    }
}

SnakeAgent* snake_agent_create(mg_arena* arena) {
    SnakeAgent* agent = MGA_PUSH_ZERO_STRUCT(arena, SnakeAgent);
    
    // 1. Create Main Network
    layer_desc descs[] = {
        { .type = LAYER_INPUT, .input = { .shape = (tensor_shape){ STATE_SIZE, 1, 1 } } },
        { .type = LAYER_DENSE, .dense = { .size = 128 } },
        { .type = LAYER_ACTIVATION, .activation = { .type = ACTIVATION_RELU } },
        { .type = LAYER_DENSE, .dense = { .size = 128 } },
        { .type = LAYER_ACTIVATION, .activation = { .type = ACTIVATION_RELU } },
        { .type = LAYER_DENSE, .dense = { .size = NUM_ACTIONS } }
    };
    
    agent->net = network_create(arena, sizeof(descs)/sizeof(layer_desc), descs, true);
    
    // 2. Create Target Network (Copy of Main)
    agent->target_net = network_create(arena, sizeof(descs)/sizeof(layer_desc), descs, false); // No training mode needed for target
    _snake_update_target_net(agent->target_net, agent->net);
    
    // 3. Init Optimizer
    agent->optim = (optimizer){
        .type = OPTIMIZER_ADAM,
        .learning_rate = 0.001f,
        .adam = { .beta1 = 0.9f, .beta2 = 0.999f, .epsilon = 1e-7f }
    };
    
    agent->epsilon = EPSILON_START;
    
    // 4. Allocate Persistent Replay Memory
    // Using arena is fine because this agent lives forever.
    agent->memory_states = MGA_PUSH_ARRAY(arena, f32, MAX_REPLAY_SIZE * STATE_SIZE);
    agent->memory_next_states = MGA_PUSH_ARRAY(arena, f32, MAX_REPLAY_SIZE * STATE_SIZE);
    
    // 5. Pre-allocate Batch Tensors
    agent->batch_states = tensor_create(arena, (tensor_shape){ STATE_SIZE, 1, BATCH_SIZE });
    agent->batch_next_states = tensor_create(arena, (tensor_shape){ STATE_SIZE, 1, BATCH_SIZE });
    
    return agent;
}

int snake_agent_act(SnakeAgent* agent, SnakeState* state, tensor* state_tensor) {
    // Epsilon Greedy
    if ((float)rand() / RAND_MAX < agent->epsilon) {
        return rand() % NUM_ACTIONS;
    }
    
    // Predict
    mga_temp scratch = mga_scratch_get(NULL, 0);
    tensor* out = tensor_create(scratch.arena, (tensor_shape){ NUM_ACTIONS, 1, 1 });
    network_feedforward(agent->net, out, state_tensor);
    
    tensor_index argmax = tensor_argmax(out);
    
    int action = argmax.y * out->shape.width + argmax.x;
    
    mga_scratch_release(scratch);
    return action;
}

void snake_agent_remember(mg_arena* arena, SnakeAgent* agent, tensor* state, int action, f32 reward, tensor* next_state, b32 done) {
    int idx = agent->replay_buffer.head;
    
    // Copy data to persistent agent memory
    memcpy(&agent->memory_states[idx * STATE_SIZE], state->data, STATE_SIZE * sizeof(f32));
    memcpy(&agent->memory_next_states[idx * STATE_SIZE], next_state->data, STATE_SIZE * sizeof(f32));
    
    Experience* exp = &agent->replay_buffer.buffer[idx];
    exp->action = action;
    exp->reward = reward;
    exp->done = done;
    
    agent->replay_buffer.head = (agent->replay_buffer.head + 1) % MAX_REPLAY_SIZE;
    if (agent->replay_buffer.count < MAX_REPLAY_SIZE) agent->replay_buffer.count++;
}

void snake_agent_train(SnakeAgent* agent) {
    if (agent->replay_buffer.count < BATCH_SIZE) return;

    mga_temp scratch = mga_scratch_get(NULL, 0);
    
    // Create temp tensors for single item processing
    tensor* state_t = tensor_create(scratch.arena, (tensor_shape){ STATE_SIZE, 1, 1 });
    tensor* next_state_t = tensor_create(scratch.arena, (tensor_shape){ STATE_SIZE, 1, 1 });
    tensor* q_eval = tensor_create(scratch.arena, (tensor_shape){ NUM_ACTIONS, 1, 1 });
    tensor* q_next = tensor_create(scratch.arena, (tensor_shape){ NUM_ACTIONS, 1, 1 });
    
    // Target tensor
    tensor* q_target = tensor_create(scratch.arena, (tensor_shape){ NUM_ACTIONS, 1, 1 });

    // Cache for backprop
    layers_cache cache = { .arena = scratch.arena };
    
    // Working tensor for feedforward/backprop
    tensor* in_out = tensor_create_alloc(scratch.arena, (tensor_shape){1,1,1}, agent->net->max_layer_size);

    for (int i = 0; i < BATCH_SIZE; i++) {
        int idx = rand() % agent->replay_buffer.count;
        
        // Copy state data from AGENT MEMORY
        memcpy(state_t->data, &agent->memory_states[idx * STATE_SIZE], STATE_SIZE * sizeof(f32));
        memcpy(next_state_t->data, &agent->memory_next_states[idx * STATE_SIZE], STATE_SIZE * sizeof(f32));
        
        // 1. Manual Feedforward State (Populating Backprop Cache)
        // Reset in_out to state input
        tensor_copy_ip(in_out, state_t);
        in_out->shape = agent->net->layers[0]->shape; 
        
        for (u32 l = 0; l < agent->net->num_layers; l++) {
            layer_feedforward(agent->net->layers[l], in_out, &cache);
        }
        // Save prediction for target calc
        tensor_copy_ip(q_eval, in_out); 

        // 2. Feedforward Next State using TARGET NETWORK
        network_feedforward(agent->target_net, q_next, next_state_t);

        // 3. Compute Target
        int action = agent->replay_buffer.buffer[idx].action;
        f32 reward = agent->replay_buffer.buffer[idx].reward;
        b32 done = agent->replay_buffer.buffer[idx].done;

        tensor_copy_ip(q_target, q_eval);
        f32* target_data = (f32*)q_target->data;
        f32* next_data = (f32*)q_next->data;

        f32 new_q = reward;
        if (!done) {
            f32 max_next = -1e9f;
            for (int a = 0; a < NUM_ACTIONS; a++) {
                if (next_data[a] > max_next) max_next = next_data[a];
            }
            new_q += GAMMA * max_next;
        }
        target_data[action] = new_q;

        // 4. Backprop
        cost_grad(COST_MEAN_SQUARED_ERROR, in_out, q_target); 
        tensor* delta = in_out; // Renamed for clarity

        for (i64 l = agent->net->num_layers - 1; l >= 0; l--) {
            layer_backprop(agent->net->layers[l], delta, &cache);
        }
    }

    // 5. Apply Changes
    for (u32 i = 0; i < agent->net->num_layers; i++) {
        layer_apply_changes(agent->net->layers[i], &agent->optim);
    }
    
    // 6. Update Target Network
    agent->train_step++;
    if (agent->train_step % TARGET_UPDATE_FREQ == 0) {
        _snake_update_target_net(agent->target_net, agent->net);
        // printf("DEBUG: Updated Target Network\n");
    }
    
    mga_scratch_release(scratch);
}

void snake_agent_decay_epsilon(SnakeAgent* agent) {
    if (agent->epsilon > EPSILON_END) {
        agent->epsilon *= EPSILON_DECAY;
    }
}

void snake_agent_save(SnakeAgent* agent, string8 path) {
    network_save(agent->net, path);
}

void snake_agent_load(SnakeAgent* agent, string8 path) {
    network_load_existing(agent->net, path);
    // Sync target net so they start equal
    _snake_update_target_net(agent->target_net, agent->net);
}
