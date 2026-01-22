#include "snake_game.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Helper to spawn food
static void _spawn_food(SnakeState* state) {
    b32 valid = false;
    while (!valid) {
        state->food.x = rand() % GRID_W;
        state->food.y = rand() % GRID_H;
        
        valid = true;
        for (int i = 0; i < state->length; i++) {
            if (state->body[i].x == state->food.x && state->body[i].y == state->food.y) {
                valid = false;
                break;
            }
        }
    }
}

void snake_init(SnakeState* state) {
    state->length = 3;
    state->body[0] = (Point){GRID_W / 2, GRID_H / 2}; // Head
    state->body[1] = (Point){GRID_W / 2, GRID_H / 2 + 1};
    state->body[2] = (Point){GRID_W / 2, GRID_H / 2 + 2}; // Tail
    
    state->score = 0;
    state->game_over = false;
    
    _spawn_food(state);
}

f32 snake_step(SnakeState* state, int action) {
    if (state->game_over) return 0.0f;
    
    Point head = state->body[0];
    
    // Move head
    if (action == ACTION_UP) head.y--;
    else if (action == ACTION_DOWN) head.y++;
    else if (action == ACTION_LEFT) head.x--;
    else if (action == ACTION_RIGHT) head.x++;
    
    // 1. Collision Check (Walls)
    if (head.x < 0 || head.x >= GRID_W || head.y < 0 || head.y >= GRID_H) {
        state->game_over = true;
        return REWARD_COLLISION;
    }
    
    // 2. Collision Check (Self)
    for (int i = 0; i < state->length; i++) {
        if (state->body[i].x == head.x && state->body[i].y == head.y) {
            state->game_over = true;
            return REWARD_COLLISION;
        }
    }
    
    // Move Body
    for (int i = state->length; i > 0; i--) {
        state->body[i] = state->body[i-1];
    }
    state->body[0] = head;
    
    // 3. Check Food
    if (head.x == state->food.x && head.y == state->food.y) {
        state->length++;
        state->score++;
        // Limit length
        if (state->length >= GRID_W * GRID_H) {
             state->length = GRID_W * GRID_H;
             state->game_over = true; // Win?
             return REWARD_FOOD * 2;
        }
        _spawn_food(state);
        return REWARD_FOOD;
    }
    
    // Decrease tail if not ate food (Wait, above logic increases length by shifting but keeping tail, correct)
    // Actually standard snake: move head, discard tail. If food, keep tail (grow).
    // My logic above: shifted everything to right. 
    // body[length] is new tail position if we didn't grow?
    // Correct logic:
    // If food: length increases, so the old tail at body[length-1] (before shift) is now body[length].
    // If no food: we just shifted, but we need to effectively 'cut' the tail?
    // Actually, simpler:
    // We already shifted everything 1 step back. 
    // If we didn't eat, we just decrement length? No, length is constant. 
    // The loop `for (int i = state->length; i > 0; i--)` moves index 0 to 1, 1 to 2...
    // So if length was 3 (0,1,2), we now have (0,1,2,3).
    // If we didn't eat, we want to discard index 3?
    // But `state->length` tracks active segments.
    // If we ate, we increment length, so index 3 becomes valid.
    // If we didn't eat, we effectively 'lost' the old tail position because we overwrote it? 
    // No, `state->body[i] = state->body[i-1]`.
    // Example length=1. body[0]=A. Loop i=1: body[1]=A. body[0]=New.
    // Result: New, A.
    // If no food, length should be 1. So we ignore body[1].
    // If food, length becomes 2. So we keep body[1].
    // Wait, the loop goes up to `state->length`. 
    // If length=3. Loop i=3: body[3] = body[2]. ... body[1] = body[0].
    // We created body[0]..body[3] (4 elements).
    // If no food, we keep length=3. So body[3] is ignored. Correct.
    
    return REWARD_STEP;
}

void snake_get_state(SnakeState* state, tensor* out) {
    tensor_fill(out, 0.0f);
    f32* data = (f32*)out->data;
    
    Point head = state->body[0];
    
    // 1. Danger Detection (Up, Right, Down, Left) - 3 levels
    // 0.0 = safe, 0.5 = body collision, 1.0 = wall
    
    // Check Up (x, y-1)
    if (head.y - 1 < 0) {
        data[0] = 1.0f;  // Wall
    } else {
        data[0] = 0.0f;  // Safe by default
        for (int i = 1; i < state->length; i++) {
            if (state->body[i].x == head.x && state->body[i].y == head.y - 1) {
                data[0] = 0.5f;  // Body
                break;
            }
        }
    }
    
    // Check Right (x+1, y)
    if (head.x + 1 >= GRID_W) {
        data[1] = 1.0f;  // Wall
    } else {
        data[1] = 0.0f;
        for (int i = 1; i < state->length; i++) {
            if (state->body[i].x == head.x + 1 && state->body[i].y == head.y) {
                data[1] = 0.5f;  // Body
                break;
            }
        }
    }
    
    // Check Down (x, y+1)
    if (head.y + 1 >= GRID_H) {
        data[2] = 1.0f;  // Wall
    } else {
        data[2] = 0.0f;
        for (int i = 1; i < state->length; i++) {
            if (state->body[i].x == head.x && state->body[i].y == head.y + 1) {
                data[2] = 0.5f;  // Body
                break;
            }
        }
    }
    
    // Check Left (x-1, y)
    if (head.x - 1 < 0) {
        data[3] = 1.0f;  // Wall
    } else {
        data[3] = 0.0f;
        for (int i = 1; i < state->length; i++) {
            if (state->body[i].x == head.x - 1 && state->body[i].y == head.y) {
                data[3] = 0.5f;  // Body
                break;
            }
        }
    }

    // 2. Current Direction (Up, Right, Down, Left)
    Point neck = state->body[1];
    if (state->length < 2) {
         data[5] = 1.0f; // Default Right
    } else {
        if (head.y < neck.y) data[4] = 1.0f; // Up
        else if (head.x > neck.x) data[5] = 1.0f; // Right
        else if (head.y > neck.y) data[6] = 1.0f; // Down
        else if (head.x < neck.x) data[7] = 1.0f; // Left
    }
    
    // 3. Food Direction (Up, Right, Down, Left)
    if (state->food.y < head.y) data[8] = 1.0f;
    if (state->food.x > head.x) data[9] = 1.0f;
    if (state->food.y > head.y) data[10] = 1.0f;
    if (state->food.x < head.x) data[11] = 1.0f;
}

void snake_render(SnakeState* state) {
    // ANSI Move to (0,0) - In-place update
    printf("\033[H"); 
    
    printf("Score:    %d\n", state->score);
    printf("Controls: WASD (Input)\n");
    
    char buffer[(GRID_W + 3) * (GRID_H + 2)];
    int buf_idx = 0;
    
    // Top Border
    for(int i=0; i<GRID_W+2; i++) buffer[buf_idx++] = '#';
    buffer[buf_idx++] = '\n';
    
    for (int y = 0; y < GRID_H; y++) {
        buffer[buf_idx++] = '#'; // Left Border
        for (int x = 0; x < GRID_W; x++) {
            char c = ' ';
            if (x == state->food.x && y == state->food.y) {
                c = '@'; // Food
            } else {
                for (int i = 0; i < state->length; i++) {
                    if (state->body[i].x == x && state->body[i].y == y) {
                        c = (i==0) ? 'O' : 'o'; // Head vs Body
                        break;
                    }
                }
            }
            buffer[buf_idx++] = c;
        }
        buffer[buf_idx++] = '#'; // Right Border
        buffer[buf_idx++] = '\n';
    }
    
    // Bottom Border
    for(int i=0; i<GRID_W+2; i++) buffer[buf_idx++] = '#';
    buffer[buf_idx++] = '\n';
    buffer[buf_idx] = '\0';
    
    printf("%s", buffer);
    fflush(stdout);
}
