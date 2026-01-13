//
// Created by Vishal Jha on 13/01/26.
//

#include "modelVariables.h"

#include <stdbool.h>

#include "../../autograd/autograd.h"
#include <stddef.h>

#include "../../memory_mngmnt/arena.h"

model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
) {
    model_var* out = PUSH_STRUCT(arena, model_var);

    out->index = model->num_vars++;
    out->flags = flags;
    out->op = MV_OP_CREATE;
    out->val = mat_create(arena, rows, cols);

    if (flags & MV_FLAG_REQUIRES_GRAD) {
        out->grad = mat_create(arena, rows, cols);
    }

    if (flags & MV_FLAG_INPUT) { model->input = out; }
    if (flags & MV_FLAG_OUTPUT) { model->output = out; }
    if (flags & MV_FLAG_DESIRED_OUTPUT) { model->desired_output = out; }
    if (flags & MV_FLAG_COST) { model->cost = out; }

    return out;
}

model_var* _mv_unary_impl(
    mem_arena* arena, model_context* model,
    model_var* input, u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    if (input->flags & MV_FLAG_REQUIRES_GRAD) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* out = mv_create(arena, model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = input;

    return out;
}

model_var* _mv_binary_impl(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b,
    u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    if (
        (a->flags & MV_FLAG_REQUIRES_GRAD) ||
        (b->flags & MV_FLAG_REQUIRES_GRAD)
    ) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* out = mv_create(arena, model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;

    return out;
}

model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
) {
    return _mv_unary_impl(
        arena, model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_RELU
    );
}

model_var* mv_softmax(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
) {
    return _mv_unary_impl(
        arena, model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_SOFTMAX
    );
}

model_var* mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_ADD
    );
}

model_var* mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_SUB
    );
}

model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a->val->cols != b->val->rows) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, b->val->cols,
        flags, MV_OP_MATMUL
    );
}

model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
) {
    if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, p, q,
        p->val->rows, p->val->cols,
        flags, MV_OP_CROSS_ENTROPY
    );
}