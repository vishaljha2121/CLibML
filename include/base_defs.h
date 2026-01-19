//
// Created by Vishal Jha on 16/01/26.
//

/**
 * @file base_defs.h
 * @brief Defines some basic types and macros
 */

#ifndef BASE_DEFS_H
#define BASE_DEFS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <assert.h>

/// 8-bit signed integer
typedef int8_t   i8;
/// 16-bit signed integer
typedef int16_t  i16;
/// 32-bit signed integer
typedef int32_t  i32;
/// 64-bit signed integer
typedef int64_t  i64;
/// 8-bit unsigned integer
typedef uint8_t  u8;
/// 16-bit unsigned integer
typedef uint16_t u16;
/// 32-bit unsigned integer
typedef uint32_t u32;
/// 64-bit unsigned integer
typedef uint64_t u64;

/// 32-bit boolean
typedef i32 b32;

/// 32-bit floating point number
typedef float  f32;
/// 64-bit floating point number
typedef double f64;

static_assert(sizeof(f32) == 4, "f32 size");
static_assert(sizeof(f64) == 8, "f64 size");

#if defined(_WIN32)
#   define PLATFORM_WIN32
#elif defined(__linux__)
#   define PLATFORM_LINUX
#endif

#ifndef THREAD_VAR
#    if defined(__clang__) || defined(__GNUC__)
#        define THREAD_VAR __thread
#    elif defined(_MSC_VER)
#        define THREAD_VAR __declspec(thread)
#    elif (__STDC_VERSION__ >= 201112L)
#        define THREAD_VAR _Thread_local
#    else
#        error "Invalid compiler/version for thread var"
#    endif
#endif

/// Marks function parameters as unused
#define UNUSED(x) (void)(x)

/// Computes the min of a and b
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
/// Computes the max of a and b
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
/// Computes the absolute value of x
#define ABS(x) ((x) < 0 ? -(x) : (x))
/// Computes the sign of x
#define SIGN(x) ((x) < 0 ? -1 : 1)

/**
 * @brief Pushes element to the front of singly linked list
 * 
 * Nodes are assumed to have a `next` member <br>
 * Does not perform any memory managment
 *
 * @param f First element of the list
 * @param l Last element of the list
 * @param n New node to be pushed
 */
#define SLL_PUSH_FRONT(f, l, n) ((f) == 0 ? \
    ((f) = (l) = (n)) :                     \
    ((n)->next = (f), (f) = (n)))           \

/**
 * @brief Pushes element to the back of singly linked list
 * 
 * Nodes are assumed to have a `next` member <br>
 * Does not perform any memory managment
 *
 * @param f First element of the list
 * @param l Last element of the list
 * @param n New node to be pushed
 */
#define SLL_PUSH_BACK(f, l, n) ((f) == 0 ? \
    ((f) = (l) = (n)) :                    \
    ((l)->next = (n), (l) = (n)),          \
    ((n)->next = 0))                       \

/**
 * @brief Removes element from front of singly linked list
 * 
 * Nodes are assumed to have a `next` member <br>
 * Does not perform any memory managment
 *
 * @param f First element of the list
 * @param l Last element of the list
 */

#define SLL_POP_FRONT(f, l) ((f) == (l) ? \
    ((f) = (l) = 0) :                     \
    ((f) = (f)->next))                    \

#endif // BASE_DEFS_H