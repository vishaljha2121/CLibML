#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "../../../include/base_defs.h"
#include "../../../include/str.h"
#include "snake_game.h"
#include "snake_ai.h"

// For sleep
#include <time.h>

void snake_train(mg_arena* arena, char* load_path) {
    printf("Initializing Snake Training...\n");
    SnakeAgent* agent = snake_agent_create(arena);
    
    if (load_path) {
        printf("Resuming training from %s\n", load_path);
        snake_agent_load(agent, str8_from_cstr((u8*)load_path));
        // Reduce epsilon if resuming, but keep some exploration
        agent->epsilon = 0.5f; 
    }
    
    SnakeState state;
    tensor* state_tensor = tensor_create(arena, (tensor_shape){ STATE_SIZE, 1, 1 });
    tensor* next_state_tensor = tensor_create(arena, (tensor_shape){ STATE_SIZE, 1, 1 });
    
    int episodes = 100000;
    
    time_t start_time = time(NULL);

    for (int e = 0; e < episodes; e++) {
        snake_init(&state);
        snake_get_state(&state, state_tensor);
        
        f32 total_reward = 0;
        int steps = 0;
        
        while (!state.game_over && steps < 100) { // Limit steps to avoid infinite loops
            int action = snake_agent_act(agent, &state, state_tensor);
            
            f32 reward = snake_step(&state, action);
            total_reward += reward;
            
            snake_get_state(&state, next_state_tensor);
            
            snake_agent_remember(arena, agent, state_tensor, action, reward, next_state_tensor, state.game_over);
            snake_agent_train(agent);
            
            // Move state pointer (Copy contents)
            memcpy(state_tensor->data, next_state_tensor->data, STATE_SIZE * sizeof(f32));
            steps++;
        }
        
        snake_agent_decay_epsilon(agent);

        if (e % 1000 == 0 || e == episodes - 1) {
            // ETA Calculation
            time_t now = time(NULL);
            double elapsed = difftime(now, start_time);
            double avg_time = (e == 0) ? 0.0 : elapsed / e;
            double remaining = avg_time * (episodes - e);
            
            int rem_h = (int)(remaining / 3600);
            int rem_m = (int)((remaining - rem_h * 3600) / 60);
            int rem_s = (int)remaining % 60;
            
            // Progress Bar
            float progress = (float)e / episodes;
            int bar_width = 40;
            int pos = bar_width * progress;
            char bar[41];
            for (int i = 0; i < bar_width; i++) {
                if (i < pos) bar[i] = '=';
                else if (i == pos) bar[i] = '>';
                else bar[i] = ' ';
            }
            bar[bar_width] = '\0';
            
            printf("\r[%-40s] %3.0f%% | Ep: %d | Score: %d | Rew: %.2f | Eps: %.3f | ETA: %02d:%02d:%02d", 
                   bar, progress * 100.0f, e, state.score, total_reward, agent->epsilon, rem_h, rem_m, rem_s);
            fflush(stdout);

            // Save model
            mga_temp scratch = mga_scratch_get(NULL, 0);
            string8 path = str8_pushf(scratch.arena, "tests/snake/snake_model_%d.tsn", e);
            snake_agent_save(agent, path);
            mga_scratch_release(scratch);
            
            if (e % 1000 == 0 && e > 0) printf("\n"); // Newline occasionally to keep history
        }
    }
    
    snake_agent_save(agent, STR8("tests/snake/snake_final.tsn"));
}

void snake_play(mg_arena* arena, char* model_path) {
    if (!model_path) {
        printf("Error: Please provide model path\n");
        return;
    }
    
    SnakeAgent* agent = snake_agent_create(arena);
    agent->epsilon = 0.0f; // No random actions
    
    printf("Loading model %s\n", model_path);
    snake_agent_load(agent, str8_from_cstr((u8*)model_path));
    
    SnakeState state;
    snake_init(&state);
    
    tensor* state_tensor = tensor_create(arena, (tensor_shape){ STATE_SIZE, 1, 1 });
    
    printf("\033[2J"); // Clear Screen
    
    while (!state.game_over) {
        snake_get_state(&state, state_tensor);
        
        int action = snake_agent_act(agent, &state, state_tensor);
        snake_step(&state, action);
        
        snake_render(&state);
        
        usleep(100000); // 100ms delay
    }
    
    printf("\nGame Over! Final Score: %d\n", state.score);
}

int snake_main(int argc, char** argv) {
    mga_desc desc = { .desired_max_size = MGA_MiB(64), .desired_block_size = MGA_MiB(4) };
    mg_arena* arena = mga_create(&desc);
    
    if (argc < 3) {
        printf("Usage: MLFramework snake [train|play] <model_path?>\n");
        return 0;
    }
    
    if (strcmp(argv[2], "train") == 0) {
        char* path = (argc > 3) ? argv[3] : NULL;
        snake_train(arena, path);
    } else if (strcmp(argv[2], "play") == 0) {
        char* path = (argc > 3) ? argv[3] : "tests/snake/snake_final.tsn";
        snake_play(arena, path);
    } else {
        printf("Unknown command: %s\n", argv[2]);
    }
    
    mga_destroy(arena);
    return 0;
}
