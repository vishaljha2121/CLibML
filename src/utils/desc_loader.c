#include "../../include/network.h"
#include "../../include/cost.h"
#include "../../include/optimizers.h"
#include <stdlib.h>

// Reusing parsing helpers if possible, or redefining them.
// ideally these should be in a common header, but for speed I will duplicate or static define.

typedef struct {
    string8 str;
    u64 index;
} _parser;

static void _parser_eat_whitespace(_parser* p) {
    while (p->index < p->str.size) {
        u8 c = p->str.str[p->index];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            p->index++;
        } else {
            break;
        }
    }
}

static b32 _parser_match_char(_parser* p, u8 c) {
    _parser_eat_whitespace(p);
    if (p->index < p->str.size && p->str.str[p->index] == c) {
        p->index++;
        return true;
    }
    return false;
}

static string8 _parser_parse_ident(mg_arena* arena, _parser* p) {
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

static u32 _parser_parse_u32(_parser* p) {
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
    string8 num_str = str8_substr(p->str, start, p->index);
    u32 out = 0;
    for (u64 i = 0; i < num_str.size; i++) {
        out = out * 10 + (num_str.str[i] - '0');
    }
    return out;
}

static f32 _parser_parse_f32(_parser* p) {
    _parser_eat_whitespace(p);
    char* end;
    f32 val = strtof((char*)(p->str.str + p->index), &end);
    p->index = (u8*)end - p->str.str;
    return val;
}

static b32 _parser_parse_bool(_parser* p) {
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

static string8 _parser_parse_string_val(mg_arena* arena, _parser* p) {
    _parser_eat_whitespace(p);
    // Parse "string" or just string until semicolon/newline
    // Assuming quoted string for paths
    if (_parser_match_char(p, '"')) {
        u64 start = p->index;
        while (p->index < p->str.size && p->str.str[p->index] != '"') {
             p->index++;
        }
        string8 out = str8_substr(p->str, start, p->index);
        _parser_match_char(p, '"');
        return out;
    }
    // fallback id
    return _parser_parse_ident(arena, p);
}

void train_desc_load(network_train_desc* out, string8 str) {
    _parser p = { .str = str, .index = 0 };
    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Apply defaults
    out->epochs = 10;
    out->batch_size = 32;
    out->optim = (optimizer){ .type = OPTIMIZER_ADAM, .learning_rate = 0.01f }; // Default optimizer
    // ...

    while (p.index < p.str.size) {
        string8 key = _parser_parse_ident(scratch.arena, &p);
        if (key.size == 0) break;
        
        if (!_parser_match_char(&p, '=')) {
              if (p.index >= p.str.size) break;
              p.index++;
              continue; // Skip invalid
        }

        if (str8_equals(key, STR8("epochs"))) out->epochs = _parser_parse_u32(&p);
        else if (str8_equals(key, STR8("batch_size"))) out->batch_size = _parser_parse_u32(&p);
        else if (str8_equals(key, STR8("learning_rate"))) {
            // out->learning_rate doesn't exist in network_train_desc, it exists in optimizer
            out->optim.learning_rate = _parser_parse_f32(&p);
        }
        // Optimizer config
        else if (str8_equals(key, STR8("optimizer"))) {
             string8 val = _parser_parse_ident(scratch.arena, &p);
             if (str8_equals(val, STR8("adam"))) out->optim.type = OPTIMIZER_ADAM;
             else if (str8_equals(val, STR8("sgd"))) out->optim.type = OPTIMIZER_SGD;
        }
        else if (str8_equals(key, STR8("save_interval"))) out->save_interval = _parser_parse_u32(&p);
        else if (str8_equals(key, STR8("save_path"))) out->save_path = _parser_parse_string_val(scratch.arena, &p);
        
        _parser_match_char(&p, ';');
    }

    mga_scratch_release(scratch);
}
