//
// Created by Vishal Jha on 13/01/26.
//

#ifndef TRAIN_H
#define TRAIN_H
#include "../variables/modelVariables.h"

void model_feedforward(model_context* model);
void model_train(
    model_context* model,
    const model_training_desc* training_desc
);
#endif //TRAIN_H
