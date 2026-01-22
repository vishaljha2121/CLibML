#ifndef SNAKE_AI_H
#define SNAKE_AI_H

#include "../../../include/network.h"
#include "../../../include/tensorNew.h"
#include "snake_game.h"

#define MAX_REPLAY_SIZE 10000
#define BATCH_SIZE 32
#define GAMMA 0.99f
#define EPSILON_START 1.0f
#define EPSILON_END 0.01f
#define EPSILON_DECAY 0.9995f // Reaches ~0.01 at 10k episodes (was 0.995 -> too fast)
#define TARGET_UPDATE_FREQ 1000 // Steps

typedef struct {
    tensor* state;
    int action;
    f32 reward;
    tensor* next_state;
    b32 done;
} Experience;

typedef struct {
    Experience buffer[MAX_REPLAY_SIZE];
    int head;
    int count;
} ReplayBuffer;

typedef struct {
    network* net;
    network* target_net; // Target Network
    
    // Persistent Replay Storage (Owned by Agent)
    f32* memory_states;      // [MAX_REPLAY_SIZE * STATE_SIZE]
    f32* memory_next_states; // [MAX_REPLAY_SIZE * STATE_SIZE]
    
    ReplayBuffer replay_buffer;
    optimizer optim;
    
    f32 epsilon;
    u64 train_step; // For target net sync
    
    // Temp tensors for training
    tensor* batch_states;
    tensor* batch_next_states;
    tensor* batch_rewards;
    int batch_actions[BATCH_SIZE];
    b32 batch_dones[BATCH_SIZE];
} SnakeAgent;

SnakeAgent* snake_agent_create(mg_arena* arena);
int snake_agent_act(SnakeAgent* agent, SnakeState* state, tensor* state_tensor);
void snake_agent_remember(mg_arena* arena, SnakeAgent* agent, tensor* state, int action, f32 reward, tensor* next_state, b32 done);
void snake_agent_train(SnakeAgent* agent);
void snake_agent_save(SnakeAgent* agent, string8 path);
void snake_agent_load(SnakeAgent* agent, string8 path);
void snake_agent_decay_epsilon(SnakeAgent* agent);

#endif // SNAKE_AI_H
