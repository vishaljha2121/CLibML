//
// Created by Vishal Jha on 13/01/26.
//

#include "modelProgram.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "../../random_generators/prng.h"
#include "../../autograd/autograd.h"

model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
) {
    mem_arena_temp scratch = arena_scratch_get(&arena, 1);

    b8* visited = PUSH_ARRAY(scratch.arena, b8, model->num_vars);

    u32 stack_size = 0;
    u32 out_size = 0;
    model_var** stack = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);
    model_var** out = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);

    stack[stack_size++] = out_var;

    while (stack_size > 0) {
        model_var* cur = stack[--stack_size];

        if (cur->index >= model->num_vars) { continue; }

        if (visited[cur->index]) {
            if (out_size < model->num_vars) {
                out[out_size++] = cur;
            }
            continue;
        }

        visited[cur->index] = true;

        if (stack_size < model->num_vars) {
            stack[stack_size++] = cur;
        }

        u32 num_inputs = MV_NUM_INPUTS(cur->op);
        for (u32 i = 0; i < num_inputs; i++) {
            model_var* input = cur->inputs[i];

            if (input->index >= model->num_vars || visited[input->index]) {
                continue;
            }

            for (u32 j = 0; j < stack_size; j++) {
                if (stack[j] == input) {
                    for (u32 k = j; k < stack_size-1; k++) {
                        stack[k] = stack[k+1];
                    }
                    stack_size--;
                }
            }

            if (stack_size < model->num_vars) {
                stack[stack_size++] = input;
            }
        }
    }

    model_program prog = {
        .size = out_size,
        .vars = PUSH_ARRAY_NZ(arena, model_var*, out_size)
    };

    memcpy(prog.vars, out, sizeof(model_var*) * out_size);

    arena_scratch_release(scratch);

    return prog;
}

void model_prog_compute(model_program* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* cur = prog->vars[i];

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        switch (cur->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE: break;

            case _MV_OP_UNARY_START: break;

            case MV_OP_RELU: { mat_relu(cur->val, a->val); } break;
            case MV_OP_SOFTMAX: { mat_softmax(cur->val, a->val); } break;

            case _MV_OP_BINARY_START: break;

            case MV_OP_ADD: { mat_add(cur->val, a->val, b->val); } break;
            case MV_OP_SUB: { mat_sub(cur->val, a->val, b->val); } break;
            case MV_OP_MATMUL: {
                mat_mul(cur->val, a->val, b->val, 1, 0, 0);
            } break;
            case MV_OP_CROSS_ENTROPY: {
                mat_cross_entropy(cur->val, a->val, b->val);
            } break;
        }
    }
}

void model_prog_compute_grads(model_program* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* cur = prog->vars[i];

        if ((cur->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
            continue;
        }

        if (cur->flags & MV_FLAG_PARAMETER) {
            continue;
        }

        mat_clear(cur->grad);
    }

    mat_fill(prog->vars[prog->size-1]->grad, 1.0f);

    for (i64 i = (i64)prog->size - 1; i >= 0; i--) {
        model_var* cur = prog->vars[i];

        if ((cur->flags & MV_FLAG_REQUIRES_GRAD) == 0) {
            continue;
        }

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        u32 num_inputs = MV_NUM_INPUTS(cur->op);

        if (
            num_inputs == 1 &&
            (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
        ) {
            continue;
        }

        if (
            num_inputs == 2 &&
            (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD &&
            (b->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
        ) {
            continue;
        }

        switch (cur->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE: break;

            case _MV_OP_UNARY_START: break;

            case MV_OP_RELU: {
                mat_relu_add_grad(a->grad, a->val, cur->grad);
            } break;
            case MV_OP_SOFTMAX: {
                mat_softmax_add_grad(a->grad, cur->val, cur->grad);
            } break;

            case _MV_OP_BINARY_START: break;

            case MV_OP_ADD: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(a->grad, a->grad, cur->grad);
                }

                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(b->grad, b->grad, cur->grad);
                }
            } break;

            case MV_OP_SUB: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(a->grad, a->grad, cur->grad);
                }

                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_sub(b->grad, b->grad, cur->grad);
                }
            } break;

            case MV_OP_MATMUL: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
                }

                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
                }
            } break;

            case MV_OP_CROSS_ENTROPY: {
                model_var* p = a;
                model_var* q = b;

                mat_cross_entropy_add_grad(
                    p->grad, q->grad, p->val, q->val, cur->grad
                );
            } break;
        }
    }
}

model_context* model_create(mem_arena* arena) {
    model_context* model = PUSH_STRUCT(arena, model_context);

    return model;
}

void model_compile(mem_arena* arena, model_context* model) {
    if (model->output != NULL) {
        model->forward_prog = model_prog_create(arena, model, model->output);
    }

    if (model->cost != NULL) {
        model->cost_prog = model_prog_create(arena, model, model->cost);
    }
}
