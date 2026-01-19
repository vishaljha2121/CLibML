//
// Created by Vishal Jha on 16/01/26.
//

/**
 * @file str.h
 * @brief Length based strings and string handling functions (based on Mr 4th's length based strings)
 *
 * This is heavily based on the string header in the [Mr 4th programming series](https://www.youtube.com/@Mr4thProgramming)
 */

#ifndef STR_H
#define STR_H

#include <stdarg.h>

#include "base_defs.h"
#include "../src/mg/mg_arena.h"

/**
 * @brief Length based 8-bit string
 * 
 * Null characters are never included
 */
typedef struct {
    /// Length of string
    u64 size;
    /// Pointer to string characters
    u8* str;
} string8;

/**
 * @brief Node of `string8_list`
 */
typedef struct string8_node {
    struct string8_node* next;
    string8 str;
} string8_node;

/**
 * @brief `string8` singly linked list
 */
typedef struct {
    string8_node* first;
    string8_node* last;

    /// NUmber of nodes
    u64 node_count;
    /// Length of all strings combined
    u64 total_size;
} string8_list; 

/**
 * @brief Creates a `string8` from a string literal
 *
 * Ex: `string8 literal = STR8("Hello World");`
 */
#define STR8(s) ((string8){ sizeof(s)-1, (u8*)s })

/**
 * @brief Creates a `string8` from the pointer range
 * 
 * Does not copy the memory
 */
string8 str8_from_range(u8* start, u8* end);
/**
 * @brief Creates a `string8` from the c string
 * 
 * Does not copy the memory
 */
string8 str8_from_cstr(u8* cstr);

/// Copies a `string8`
string8 str8_copy(mg_arena* arena, string8 str);
/// Creates a c string from a `string8`
u8* str8_to_cstr(mg_arena* arena, string8 str);

/// Returns true if `a` and `b` are equal
b32 str8_equals(string8 a, string8 b);
/// Returns true if `sub` appears in `str`
b32 str8_contains(string8 str, string8 sub);
/// Returns true if `c` appears `str`
b32 str8_contains_char(string8 str, u8 c);

/**
 * @brief Gets the index of the first occurrence of `sub` in `str`
 *
 * @param index Index of the first occurrence of `sub` in `str`
 *  is put in this pointer if an occurrence exists
 *
 * @return true if `sub` is in `str`, false otherwise
 */
b32 str8_index_of(string8 str, string8 sub, u64* index);
/**
 * @brief Gets the index of the first occurrence of `c` in `str`
 *
 * @param index Index of the first occurrence of `c` in `str`
 *  is put in this pointer if an occurrence exists
 *
 * @return true if `c` is in `str`, false otherwise
 */
b32 str8_index_of_char(string8 str, u8 c, u64* index);

/// Creates a `string8` that points to the substring in `str` (does not copy memory)
string8 str8_substr(string8 str, u64 start, u64 end);
/// Creates a `string8` that points to the substring in `str` (does not copy memory)
string8 str8_substr_size(string8 str, u64 start, u64 size);

/// Creates a new `string8` without any occurrences of  ' ', '\t', '\n', and '\r'
string8 str8_remove_space(mg_arena* arena, string8 str);

/**
 * @brief Pushes a `string8` to the `string8_list` with an already allocated node
 *
 * @param list String list to push to
 * @param str String to push
 * @param node Allocated node that is being pushed
 */
void str8_list_push_existing(string8_list* list, string8 str, string8_node* node);
/// Pushes a `string8` to the `string8_list` while allocating the node
void str8_list_push(mg_arena* arena, string8_list* list, string8 str);

/// Creates a string that joins together all of the strings in `list`
string8 str8_concat(mg_arena* arena, string8_list list);

/**
 * @brief Creates a formated `string8` from a c string and a `va_list`
 *
 * Formats string according to the c string format system (i.e. printf) <br>
 * See `str8_pushf` for not `va_list` version
 *
 * @param arena Arena to allocate `string8` on
 * @param fmt C string with specifying the format (e.g. `"Num: %u"`)
 * @param args List of arguments for format
 */
string8 str8_pushfv(mg_arena* arena, const char* fmt, va_list args);
/**
 * @brief Creates a formated `string8` from a c string and a list of arguments
 *
 * Formats string according to the c string format system (i.e. printf) <br>
 *
 * @param arena Arena to allocate `string8` on
 * @param fmt C string with specifying the format (e.g. `"Num: %u"`)
 */
string8 str8_pushf(mg_arena* arena, const char* fmt, ...);

#endif // STR_H