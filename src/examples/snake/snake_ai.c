#include "snake_ai.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

static f32* g_state_storage = NULL;
static f32* g_next_state_storage = NULL;

SnakeAgent* snake_agent_create(mg_arena* arena) {
    SnakeAgent* agent = MGA_PUSH_ZERO_STRUCT(arena, SnakeAgent);
    
    // 1. Create Network
    // Input: GRID_W * GRID_H (Flattened grid)
    // Output: 4 Actions
    layer_desc descs[] = {
        { .type = LAYER_INPUT, .input = { .shape = (tensor_shape){ GRID_W * GRID_H, 1, 1 } } },
        { .type = LAYER_DENSE, .dense = { .size = 128 } },
        { .type = LAYER_ACTIVATION, .activation = { .type = ACTIVATION_RELU } },
        { .type = LAYER_DENSE, .dense = { .size = 128 } },
        { .type = LAYER_ACTIVATION, .activation = { .type = ACTIVATION_RELU } },
        { .type = LAYER_DENSE, .dense = { .size = NUM_ACTIONS } }
        // No Softmax for Q-Values! We want raw scores.
    };
    
    agent->net = network_create(arena, sizeof(descs)/sizeof(layer_desc), descs, true);
    
    // 2. Init Optimizer
    agent->optim = (optimizer){
        .type = OPTIMIZER_ADAM,
        .learning_rate = 0.001f,
        .adam = { .beta1 = 0.9f, .beta2 = 0.999f, .epsilon = 1e-7f }
    };
    
    agent->epsilon = EPSILON_START;
    
    // 3. Pre-allocate Batch Tensors
    agent->batch_states = tensor_create(arena, (tensor_shape){ GRID_W * GRID_H, 1, BATCH_SIZE });
    agent->batch_next_states = tensor_create(arena, (tensor_shape){ GRID_W * GRID_H, 1, BATCH_SIZE });
    
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
    
    // We need to COPY tensors because 'state' might be temporary or changing
    // Usually memory management for RL is tricky in arenas.
    // For this example, we'll assume we copy 
    // BUT our arena strategy is linear. We can't just keep allocating forever.
    // Ideally Replay Buffer uses a separate arena that resets? Or we specific persistent allocator?
    // Given the constraints: Let's assume the state tensors passed are persistent or we copy them to a specific buffer area.
    // For simplicity: We will just MALLOC the data for buffer storage to avoid arena overflow, or assume `arena` passed is a long-lived one.
    // Let's alloc fresh copies.
    
    Experience* exp = &agent->replay_buffer.buffer[idx];
    
    // If wrapping around, we should free old ones? 
    // Arena doesn't support free.
    // This is a limitation of linear arenas for Ring Buffers.
    // WORKAROUND: Alloc 'state_data' once equal to MAX_REPLAY_SIZE * StateSize.
    // Just copy DATA.
    // We need a persistent storage for replay buffer data.
    // For now, let's just use `malloc/free` for the replay buffer content for safety in this specific example, 
    // OR just use a fixed large array if size is small.
    // 10x10 = 100 floats. 10k items = 1M floats = 4MB. Small enough.
    
    // Use global storage
    if (!g_state_storage) {
        g_state_storage = malloc(MAX_REPLAY_SIZE * GRID_W * GRID_H * sizeof(f32));
        g_next_state_storage = malloc(MAX_REPLAY_SIZE * GRID_W * GRID_H * sizeof(f32));
        
        // Zero init
        memset(g_state_storage, 0, MAX_REPLAY_SIZE * GRID_W * GRID_H * sizeof(f32));
        memset(g_next_state_storage, 0, MAX_REPLAY_SIZE * GRID_W * GRID_H * sizeof(f32));
    }
    
    // Copy data
    memcpy(&g_state_storage[idx * GRID_W * GRID_H], state->data, GRID_W * GRID_H * sizeof(f32));
    memcpy(&g_next_state_storage[idx * GRID_W * GRID_H], next_state->data, GRID_W * GRID_H * sizeof(f32));
    
    // Create 'View' Tensors (Don't own data, just point to storage)
    // Actually we can't easily create view tensors without creating a struct.
    // We will create temp tensors during training from this data.
    
    exp->action = action;
    exp->reward = reward;
    exp->done = done;
    
    agent->replay_buffer.head = (agent->replay_buffer.head + 1) % MAX_REPLAY_SIZE;
    if (agent->replay_buffer.count < MAX_REPLAY_SIZE) agent->replay_buffer.count++;
}
// Helper to access stored data
static f32* _get_state_ptr(int idx) {
    // Only works because of the static allocation above. 
    // Not thread safe or multi-instance safe, but fine for CLI demo.
    // Ideally this pointer would be in SnakeAgent struct.
    // Re-accessing the static pointers is tricky. 
    // Let's assume the user calls helper.
    // Wait, let's move storage to SnakeAgent to be clean.
    return NULL; // Can't access static. 
}

// Redefining remember to use agent-owned memory
// We need to add fields to SnakeAgent in header? Too late, header writen.
// We can use the 'extra' memory pattern or just modify the header.
// Let's modify the implementation to use a lazy-init static if needed, 
// OR simpler: use `tensor_copy` using the provided arena, assuming the user resets arena occasionally?
// No, replay buffer persists across episodes.
// Let's stick to the static implementation for simplicity but make the pointers file-scope globals.


// Unused incomplete implementation removed
void snake_agent_train(SnakeAgent* agent) {
    if (agent->replay_buffer.count < BATCH_SIZE) return;
    if (!g_state_storage) return; 

    mga_temp scratch = mga_scratch_get(NULL, 0);
    
    // Create temp tensors for single item processing
    tensor* state_t = tensor_create(scratch.arena, (tensor_shape){ GRID_W * GRID_H, 1, 1 });
    tensor* next_state_t = tensor_create(scratch.arena, (tensor_shape){ GRID_W * GRID_H, 1, 1 });
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
        
        // Copy state data
        memcpy(state_t->data, &g_state_storage[idx * GRID_W * GRID_H], GRID_W * GRID_H * sizeof(f32));
        memcpy(next_state_t->data, &g_next_state_storage[idx * GRID_W * GRID_H], GRID_W * GRID_H * sizeof(f32));
        
        // 1. Manual Feedforward State (Populating Backprop Cache)
        // Reset in_out to state input
        tensor_copy_ip(in_out, state_t);
        // Ensure input shape is correct logic for layer 0 (though layer_feedforward handles it usually)
        in_out->shape = agent->net->layers[0]->shape; 
        
        for (u32 l = 0; l < agent->net->num_layers; l++) {
            layer_feedforward(agent->net->layers[l], in_out, &cache);
        }
        // Save prediction for target calc
        tensor_copy_ip(q_eval, in_out); 

        // 2. Feedforward Next State (No Cache, using standard function is fine as it uses temp scratch)
        network_feedforward(agent->net, q_next, next_state_t);

        // 3. Compute Target
        int action = agent->replay_buffer.buffer[idx].action;
        f32 reward = agent->replay_buffer.buffer[idx].reward;
        b32 done = agent->replay_buffer.buffer[idx].done;

        // Copy q_eval to target then modify specific action
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
        // cost_grad computes (in_out - desired) -> delta.
        // in_out currently holds q_eval result.
        // We calculate Gradient: (Prediction - Target)
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
    
    mga_scratch_release(scratch);
}

void snake_agent_decay_epsilon(SnakeAgent* agent) {
    if (agent->epsilon > EPSILON_END) {
        agent->epsilon *= EPSILON_DECAY;
    }
}

// function removed

void snake_agent_save(SnakeAgent* agent, string8 path) {
    network_save(agent->net, path);
}

void snake_agent_load(SnakeAgent* agent, string8 path) {
    // We need a temp arena for loading? 
    // `network_load` creates a CURRENT arena copy. 
    // We want to load into our EXISTING agent->net?
    // `network_load` creates a NEW network.
    // We can just replace the pointer.
    
    // Ideally we should delete old one, but we are in a linear arena...
    // We can just overwrite.
    mga_temp scratch = mga_scratch_get(NULL, 0);
    network* loaded = network_load(scratch.arena, path, true); // Load to scratch? No, it needs to persist.
    
    // network_load allocates on the passed arena.
    // We passed scratch. So it will die.
    // We need to pass agent's arena? We don't have access to it here easily without storing it in Agent.
    // BUT `network.h` says `network_load_existing`. 
    // "void network_load_existing(network* nn, string8 file_name);"
    // This allows loading params into existing structure! Perfect.
    
    network_load_existing(agent->net, path);
    mga_scratch_release(scratch);
}
