//
// Created by Vishal Jha on 13/01/26.
//

#ifndef MODELPROGRAM_H
#define MODELPROGRAM_H
#include "../variables/modelVariables.h"


model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
);
void model_prog_compute(model_program* prog);
void model_prog_compute_grads(model_program* prog);

model_context* model_create(mem_arena* arena);
void model_compile(mem_arena* arena, model_context* model);
#endif //MODELPROGRAM_H
