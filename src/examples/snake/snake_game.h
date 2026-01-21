#ifndef SNAKE_GAME_H
#define SNAKE_GAME_H

#include "../../../include/base_defs.h"
#include "../../../include/tensorNew.h"

// Grid Size
#define GRID_W 10
#define GRID_H 10

// Actions
#define ACTION_UP 0
#define ACTION_DOWN 1
#define ACTION_LEFT 2
#define ACTION_RIGHT 3
#define NUM_ACTIONS 4

// Rewards
#define REWARD_FOOD 20.0f
#define REWARD_COLLISION -100.0f
#define REWARD_STEP -0.1f

typedef struct {
    int x, y;
} Point;

typedef struct {
    Point body[GRID_W * GRID_H];
    int length;
    Point food;
    int score;
    b32 game_over;
} SnakeState;

// Initialize game
void snake_init(SnakeState* state);

// Step function: Takes action, updates state, returns reward
// done flag is set in state->game_over
f32 snake_step(SnakeState* state, int action);

// Convert state to tensor input (Grid representation)
// Output tensor shape: (GRID_W, GRID_H, 1) or flattened
void snake_get_state(SnakeState* state, tensor* out);

// Render game to console (Retro style in-place)
void snake_render(SnakeState* state);

#endif // SNAKE_GAME_H
