//
// Created by Vishal Jha on 16/01/26.
//
#include "../include/err.h"

#include <stdio.h>

static const string8 _error_strings[ERR_COUNT] = {
#define X(code) { (u64)sizeof(#code), (u8*)#code },
    ERROR_XLIST
#undef X
};

static void _default_error_callback(error err) {
    string8 code_str = err_to_str(err.code);

    fprintf(stderr, "TurboSpork %.*s: \"%.*s\"\n", (int)code_str.size, (char*)code_str.str, (int)err.msg.size, (char*)err.msg.str);
}

static error_callback* _error_callback = _default_error_callback;

void err(error err) {
    _error_callback(err);
}

void err_set_callback(error_callback* callback) {
    _error_callback = callback;
}

string8 err_to_str(error_code code) {
    if (code >= ERR_COUNT) {
        return _error_strings[0];
    }

    return _error_strings[code];
}
error_code err_from_str(string8 str) {
    error_code out = 0;

    for (u32 i = 0; i < ERR_COUNT; i++) {
        if (str8_equals(str, _error_strings[i])) {
            out = i;
            break;
        }
    }

    return out;
}