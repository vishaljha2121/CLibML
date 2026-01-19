//
// Created by Vishal Jha on 16/01/26.
//
#include "../../include/layers.h"
#include "layers_internal.h"
#include "../../include/err.h"
#include "../random_generators/prng.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static _layer_func_defs _layer_funcs[LAYER_COUNT] = {
    [LAYER_NULL] = {
        .create = _layer_null_create,
        .feedforward = _layer_null_feedforward,
        .backprop = _layer_null_backprop,
        .apply_changes = _layer_null_apply_changes,
        .delete = _layer_null_delete,
        .save = _layer_null_save,
        .load = _layer_null_load
    },
    [LAYER_INPUT] = {
        .create = _layer_input_create,
        .feedforward = _layer_input_feedforward,
    },
    [LAYER_RESHAPE] = {
        .create = _layer_reshape_create,
        .feedforward = _layer_reshape_feedforward,
        .backprop = _layer_reshape_backprop,
    },
    [LAYER_DENSE] = {
        .create = _layer_dense_create,
        .feedforward = _layer_dense_feedforward,
        .backprop = _layer_dense_backprop,
        .apply_changes = _layer_dense_apply_changes,
        .delete = _layer_dense_delete,
        .save = _layer_dense_save,
        .load = _layer_dense_load,
    },
    [LAYER_ACTIVATION] = {
        .create = _layer_activation_create,
        .feedforward = _layer_activation_feedforward,
        .backprop = _layer_activation_backprop,
    },
    [LAYER_DROPOUT] = {
        .create = _layer_dropout_create,
        .feedforward = _layer_dropout_feedforward,
        .backprop = _layer_dropout_backprop,
    },
    [LAYER_FLATTEN] = {
        .create = _layer_flatten_create,
        .feedforward = _layer_flatten_feedforward,
        .backprop = _layer_flatten_backprop,
    },
    [LAYER_POOLING_2D] = {
        .create = _layer_pooling_2d_create,
        .feedforward = _layer_pooling_2d_feedforward,
        .backprop = _layer_pooling_2d_backprop,
    },
    [LAYER_CONV_2D] = {
        .create = _layer_conv_2d_create,
        .feedforward = _layer_conv_2d_feedforward,
        .backprop = _layer_conv_2d_backprop,
        .apply_changes = _layer_conv_2d_apply_changes,
        .delete = _layer_conv_2d_delete,
        .save = _layer_conv_2d_save,
        .load = _layer_conv_2d_load,
    },
    [LAYER_NORM] = {
        .create = _layer_norm_create,
        .feedforward = _layer_norm_feedforward,
        .backprop = _layer_norm_backprop,
    }
};

string8 layer_get_name(layer_type type) {
    switch (type) {
        case LAYER_NULL: return STR8("null");
        case LAYER_INPUT: return STR8("input");
        case LAYER_RESHAPE: return STR8("reshape");
        case LAYER_DENSE: return STR8("dense");
        case LAYER_ACTIVATION: return STR8("activation");
        case LAYER_DROPOUT: return STR8("dropout");
        case LAYER_FLATTEN: return STR8("flatten");
        case LAYER_POOLING_2D: return STR8("pooling_2d");
        case LAYER_CONV_2D: return STR8("conv_2d");
        case LAYER_NORM: return STR8("norm");
        default: return STR8("unknown");
    }
}

layer_type layer_from_name(string8 name) {
    if (str8_equals(name, STR8("null"))) return LAYER_NULL;
    if (str8_equals(name, STR8("input"))) return LAYER_INPUT;
    if (str8_equals(name, STR8("reshape"))) return LAYER_RESHAPE;
    if (str8_equals(name, STR8("dense"))) return LAYER_DENSE;
    if (str8_equals(name, STR8("activation"))) return LAYER_ACTIVATION;
    if (str8_equals(name, STR8("dropout"))) return LAYER_DROPOUT;
    if (str8_equals(name, STR8("flatten"))) return LAYER_FLATTEN;
    if (str8_equals(name, STR8("pooling_2d"))) return LAYER_POOLING_2D;
    if (str8_equals(name, STR8("conv_2d"))) return LAYER_CONV_2D;
    if (str8_equals(name, STR8("norm"))) return LAYER_NORM;
    return LAYER_NULL;
}

layer* layer_create(mg_arena* arena, const layer_desc* desc, tensor_shape prev_shape) {
    if (desc->type >= LAYER_COUNT) {
        ERR(ERR_INVALID_INPUT, "Invalid layer type");
        return NULL;
    }
    
    layer* out = MGA_PUSH_ZERO_STRUCT(arena, layer);
    out->type = desc->type;
    out->training_mode = desc->training_mode;

    if (_layer_funcs[out->type].create) {
        _layer_funcs[out->type].create(arena, out, desc, prev_shape);
    }
    
    return out;
}

void layer_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    if (l->type >= LAYER_COUNT) return;
    if (_layer_funcs[l->type].feedforward) {
        _layer_funcs[l->type].feedforward(l, in_out, cache);
    }
}

void layer_backprop(layer* l, tensor* delta, layers_cache* cache) {
    if (l->type >= LAYER_COUNT) return;
    if (_layer_funcs[l->type].backprop) {
        _layer_funcs[l->type].backprop(l, delta, cache);
    }
}

void layer_apply_changes(layer* l, const optimizer* optim) {
    if (l->type >= LAYER_COUNT) return;
    if (_layer_funcs[l->type].apply_changes) {
        _layer_funcs[l->type].apply_changes(l, optim);
    }
}

void layer_delete(layer* l) {
    if (l->type >= LAYER_COUNT) return;
    if (_layer_funcs[l->type].delete) {
        _layer_funcs[l->type].delete(l);
    }
}

void layer_save(mg_arena* arena, layer* l, tensor_list* list, u32 index) {
    if (l->type >= LAYER_COUNT) return;
    if (_layer_funcs[l->type].save) {
        _layer_funcs[l->type].save(arena, l, list, index);
    }
}

void layer_load(layer* l, const tensor_list* list, u32 index) {
    if (l->type >= LAYER_COUNT) return;
    if (_layer_funcs[l->type].load) {
        _layer_funcs[l->type].load(l, list, index);
    }
}

layer_desc layer_desc_default(layer_type type) {
    layer_desc out = { .type = type };
    switch (type) {
        case LAYER_DENSE:
            out.dense.weight_init = PARAM_INIT_XAVIER_UNIFORM;
            out.dense.bias_init = PARAM_INIT_ZEROS;
            break;
        case LAYER_CONV_2D:
            out.conv_2d.stride = 1;
            out.conv_2d.kernels_init = PARAM_INIT_HE_NORMAL;
            out.conv_2d.biases_init = PARAM_INIT_ZEROS;
            break;
        case LAYER_POOLING_2D:
            out.pooling_2d.type = POOLING_MAX;
            break;
        case LAYER_ACTIVATION:
            out.activation.type = ACTIVATION_RELU;
            break;
        default: break;
    }
    return out;
}

layer_desc layer_desc_apply_default(const layer_desc* desc) {
    layer_desc out = *desc;
    layer_desc def = layer_desc_default(desc->type);
    
    switch (desc->type) {
        case LAYER_DENSE:
            if (out.dense.weight_init == 0) out.dense.weight_init = def.dense.weight_init;
            if (out.dense.bias_init == 0) out.dense.bias_init = def.dense.bias_init;
            break;
        case LAYER_CONV_2D:
            if (out.conv_2d.stride == 0) out.conv_2d.stride = def.conv_2d.stride;
            if (out.conv_2d.kernels_init == 0) out.conv_2d.kernels_init = def.conv_2d.kernels_init;
            if (out.conv_2d.biases_init == 0) out.conv_2d.biases_init = def.conv_2d.biases_init;
            break;
        case LAYER_POOLING_2D:
            if (out.pooling_2d.type == 0) out.pooling_2d.type = def.pooling_2d.type;
            break;
        case LAYER_ACTIVATION:
            if (out.activation.type == 0) out.activation.type = def.activation.type;
            break;
        default: break;
    }
    return out;
}

void layer_desc_save(mg_arena* arena, string8_list* list, const layer_desc* desc) {
    string8 header = str8_pushf(arena, "%s: ", layer_get_name(desc->type).str);
    str8_list_push(arena, list, header);

    switch (desc->type) {
        case LAYER_INPUT:
            str8_list_push(arena, list, str8_pushf(arena, "shape = (%u, %u, %u); ", desc->input.shape.width, desc->input.shape.height, desc->input.shape.depth));
            break;
        case LAYER_RESHAPE:
            str8_list_push(arena, list, str8_pushf(arena, "shape = (%u, %u, %u); ", desc->reshape.shape.width, desc->reshape.shape.height, desc->reshape.shape.depth));
            break;
        case LAYER_DENSE:
            str8_list_push(arena, list, str8_pushf(arena, "size = %u; ", desc->dense.size));
            break;
        case LAYER_ACTIVATION:
            // TODO: map enum to string
            str8_list_push(arena, list, str8_pushf(arena, "type = %d; ", desc->activation.type));
            break;
        case LAYER_DROPOUT:
            str8_list_push(arena, list, str8_pushf(arena, "keep_rate = %f; ", desc->dropout.keep_rate));
            break;
        // Add other layers as needed
        default:
             str8_list_push(arena, list, STR8("; "));
             break;
    }
     // Newline for readability
    str8_list_push(arena, list, STR8("\n"));
}

// Parsing helpers
typedef struct {
    string8 str;
    u64 index;
} _parser;

void _parser_eat_whitespace(_parser* p) {
    while (p->index < p->str.size) {
        u8 c = p->str.str[p->index];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            p->index++;
        } else {
            break;
        }
    }
}

b32 _parser_match_char(_parser* p, u8 c) {
    _parser_eat_whitespace(p);
    if (p->index < p->str.size && p->str.str[p->index] == c) {
        p->index++;
        return true;
    }
    return false;
}

string8 _parser_parse_ident(mg_arena* arena, _parser* p) {
    _parser_eat_whitespace(p);
    u64 start = p->index;
    while (p->index < p->str.size) {
        u8 c = p->str.str[p->index];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
            p->index++;
        } else {
            break;
        }
    }
    return str8_substr(p->str, start, p->index);
}


u32 _parser_parse_u32(_parser* p) {
    _parser_eat_whitespace(p);
    u64 start = p->index;
    while (p->index < p->str.size) {
        u8 c = p->str.str[p->index];
        if (c >= '0' && c <= '9') {
            p->index++;
        } else {
            break;
        }
    }
    
    // Simple u32 parsing
    string8 num_str = str8_substr(p->str, start, p->index);
    u32 out = 0;
    // VERY Basic Atoi... could be better but sufficient for now
    for (u64 i = 0; i < num_str.size; i++) {
        out = out * 10 + (num_str.str[i] - '0');
    }
    return out;
}

f32 _parser_parse_f32(_parser* p) {
    _parser_eat_whitespace(p);
    char* end;
    f32 val = strtof((char*)(p->str.str + p->index), &end);
    p->index = (u8*)end - p->str.str;
    return val;
}

b32 _parser_parse_bool(_parser* p) {
    _parser_eat_whitespace(p);
    if (p->index + 4 <= p->str.size && strncmp((char*)(p->str.str + p->index), "true", 4) == 0) {
        p->index += 4;
        return true;
    }
    if (p->index + 5 <= p->str.size && strncmp((char*)(p->str.str + p->index), "false", 5) == 0) {
        p->index += 5;
        return false;
    }
    return false;
}
b32 _parser_parse_tensor_shape(_parser* p, tensor_shape* out) {
    if (!_parser_match_char(p, '(')) return false;
    out->width = _parser_parse_u32(p);
    if (!_parser_match_char(p, ',')) return false;
    out->height = _parser_parse_u32(p);

    if (_parser_match_char(p, ',')) {
         out->depth = _parser_parse_u32(p);
    } else {
        out->depth = 1;
    }
    
    // Optional trailing comma
    _parser_match_char(p, ',');
    
    return _parser_match_char(p, ')');
}

b32 layer_desc_load(layer_desc* out, string8 str) {
    // 1. Parse Type
    // Format: "type:\n key = value;\n ..."
    
    _parser p = { .str = str, .index = 0 };
    
    // We expect the type to be at the beginning, followed by a colon
    // BUT the passed string might be "type: ..." OR just the content string if headers are peeled?
    // looking at `network.c`, it splits by semi-colons, then passes the substring.
    // wait, `network.c` splits by semi-colons.
    u64 colon_index = 0;
    if (!str8_index_of_char(str, ':', &colon_index)) return false;

    string8 type_str = str8_substr(str, 0, colon_index);
    mga_temp scratch = mga_scratch_get(NULL, 0);

    type_str = str8_remove_space(scratch.arena, type_str);
    out->type = layer_from_name(type_str);
    
    // Advance parser to after colon
    p.index = colon_index + 1;
    
    // Apply defaults first so partial config works
    *out = layer_desc_apply_default(out);

    // Loop through fields
    while (p.index < p.str.size) {
        string8 key = _parser_parse_ident(scratch.arena, &p);
        if (key.size == 0) break; // End of string or invalid
        
        // Check for colon (which might mean we overshot into the next layer if semicolor was missed?)
        // Or check for equals
        if (!_parser_match_char(&p, '=')) {
             // If we hit a semicolon without an equals, maybe it was an empty statement or end of block?
             if (_parser_match_char(&p, ';')) continue;
             
             // If we didn't find =, and it wasn't a semicolon, break to avoid infinite loop
             // break; 
             // actually, might be end of string
             if (p.index >= p.str.size) break;
             
             // consume one char to avoid infinite loop on error
             p.index++;
             continue;
        }
        
        // Parse Value based on key/type
        if (out->type == LAYER_CONV_2D) {
            if (str8_equals(key, STR8("num_filters"))) out->conv_2d.num_filters = _parser_parse_u32(&p);
            else if (str8_equals(key, STR8("kernel_size"))) out->conv_2d.kernel_size = _parser_parse_u32(&p);
            else if (str8_equals(key, STR8("padding"))) out->conv_2d.padding = _parser_parse_bool(&p);
            else if (str8_equals(key, STR8("stride"))) out->conv_2d.stride = _parser_parse_u32(&p);
        } else if (out->type == LAYER_DENSE) {
            if (str8_equals(key, STR8("size"))) out->dense.size = _parser_parse_u32(&p);
        } else if (out->type == LAYER_ACTIVATION) {
             if (str8_equals(key, STR8("type"))) {
                 string8 val = _parser_parse_ident(scratch.arena, &p);
                 if (str8_equals(val, STR8("relu"))) out->activation.type = ACTIVATION_RELU;
                 else if (str8_equals(val, STR8("sigmoid"))) out->activation.type = ACTIVATION_SIGMOID;
                 else if (str8_equals(val, STR8("tanh"))) out->activation.type = ACTIVATION_TANH;
                 else if (str8_equals(val, STR8("softmax"))) out->activation.type = ACTIVATION_SOFTMAX;
                 else if (str8_equals(val, STR8("linear"))) out->activation.type = ACTIVATION_LINEAR;
             }
        } else if (out->type == LAYER_POOLING_2D) {
            if (str8_equals(key, STR8("pool_size"))) _parser_parse_tensor_shape(&p, &out->pooling_2d.pool_size);
            else if (str8_equals(key, STR8("type"))) {
                 string8 val = _parser_parse_ident(scratch.arena, &p);
                 if (str8_equals(val, STR8("max"))) out->pooling_2d.type = POOLING_MAX;
                 else if (str8_equals(val, STR8("avg"))) out->pooling_2d.type = POOLING_AVG;
            }
        } else if (out->type == LAYER_INPUT || out->type == LAYER_RESHAPE) {
            // Note: input/reshape use the same struct layout for shape
            if (str8_equals(key, STR8("shape"))) _parser_parse_tensor_shape(&p, &out->input.shape);
        } else if (out->type == LAYER_DROPOUT) {
             if (str8_equals(key, STR8("keep_rate"))) out->dropout.keep_rate = _parser_parse_f32(&p);
        }
        
        //     ...
        //
        // If `_network_load_layout_impl` splits by semicolon, then `layer_desc_load` only gets "num_filters = 32" if it was split there.
        // BUT the user prompts shows block structure.
        // Let's re-read `network.c`: `_network_load_layout_impl`
        /*
        for (u64 i = 0; i < file.size; i++) {
             u8 c = file.str[i];
             if (c == ';') { ... continue; } // Semi-colon ends a layer desc?
             if (c == ':') { ... } // Colon triggers start of NEW desc? 
        }
        */
        // The implementation in `network.c` seems to assume that ':' marks the END of the previous identifier (the type) which effectively starts a new block. 
        // And it accumulates the string until the NEXT ':'?
        // Actually: `string8 desc_str = str8_substr(file, desc_str_start, last_semi + 1);`
        // It seems it relies on semicolons as delimiters for LAYERS too? or maybe just fields?
        
        // Critical: The user example provided in the prompt implies:
        // `conv_2d: num_filters=32; ...` 
        // If `network.c` splits by `:` to find types...
        // `network.c` logic:
        // Loop chars.
        // if ';', last_semi = i.
        // if ':', push previous string (desc_str_start to last_semi+1). update desc_str_start.
        // This implies that the string passed to `layer_desc_load` contains everything from the start of the identifier (e.g. "conv_2d") up to the last semicolon of that block.
        
        // So `p` here will contain "num_filters = 32; kernel_size = 3; ..."
        // It's my job here to parse multiple key-values if they are in the string.
        
        _parser_match_char(&p, ';');
    }

    mga_scratch_release(scratch);
    return true;
}

void param_init(tensor* param, param_init_type input_type, u64 in_size, u64 out_size) {
    switch (input_type) {
        case PARAM_INIT_ZEROS:
            tensor_fill(param, 0.0f);
            break;
        case PARAM_INIT_ONES:
            tensor_fill(param, 1.0f);
            break;
        case PARAM_INIT_XAVIER_UNIFORM: {
            f32 scale = sqrtf(6.0f / (f32)(in_size + out_size));
            u64 size = (u64)param->shape.width * param->shape.height * param->shape.depth;
            f32* data_ptr = (f32*)param->data;
            for (u64 i = 0; i < size; i++) {
                data_ptr[i] = (prng_rand_f32() * 2.0f - 1.0f) * scale;
            }
        } break;
        case PARAM_INIT_XAVIER_NORMAL: {
            f32 scale = sqrtf(2.0f / (f32)(in_size + out_size));
            u64 size = (u64)param->shape.width * param->shape.height * param->shape.depth;
            f32* data_ptr = (f32*)param->data;
            for (u64 i = 0; i < size; i++) {
                f32 u1 = prng_rand_f32();
                f32 u2 = prng_rand_f32();
                f32 z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
                data_ptr[i] = z * scale;
            }
        } break;
        case PARAM_INIT_HE_UNIFORM: {
            f32 scale = sqrtf(6.0f / (f32)in_size);
            u64 size = (u64)param->shape.width * param->shape.height * param->shape.depth;
            f32* data_ptr = (f32*)param->data;
            for (u64 i = 0; i < size; i++) {
                data_ptr[i] = (prng_rand_f32() * 2.0f - 1.0f) * scale;
            }
        } break;
        case PARAM_INIT_HE_NORMAL: {
            f32 scale = sqrtf(2.0f / (f32)in_size);
            u64 size = (u64)param->shape.width * param->shape.height * param->shape.depth;
            f32* data_ptr = (f32*)param->data;
            for (u64 i = 0; i < size; i++) {
                f32 u1 = prng_rand_f32();
                f32 u2 = prng_rand_f32();
                f32 z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
                data_ptr[i] = z * scale;
            }
        } break;
        default: break;
    }
}

void layers_cache_push(layers_cache* cache, tensor* t) {
    layers_cache_node* node = MGA_PUSH_ZERO_STRUCT(cache->arena, layers_cache_node);
    node->t = t;
    
    // Stack Push (Front)
    node->next = cache->first;
    cache->first = node;
    
    if (cache->last == NULL) {
        cache->last = node;
    }
}

tensor* layers_cache_pop(layers_cache* cache) {
    if (cache->first == NULL) {
        return NULL;
    }
    
    // Stack Pop (Front)
    layers_cache_node* node = cache->first;
    cache->first = node->next;
    
    if (cache->first == NULL) {
        cache->last = NULL;
    }
    
    return node->t;
}