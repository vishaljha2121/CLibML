//
// Created by Vishal Jha on 16/01/26.
//

/**
 * @file os.h
 * @brief Operating system specific functions
 */

#ifndef OS_H
#define OS_H

#include "base_defs.h"
#include "str.h"
#include "../src/mg/mg_arena.h"

/// File flags
typedef enum {
    /// File is directory
    FILE_IS_DIR = (1 << 0)
} file_flags;

// TODO: make this consistent between linux and windows
// TODO: confirm bounds here

/// Stores date and time
typedef struct {
    /**
     * @brief Seconds [0, 60]
     *
     * 60 is included because of leap seconds
     */
    u8 sec;
    /// Minutes [0, 59]
    u8 min;
    /// Hour [0, 23]
    u8 hour;
    /// Day [1, 31]
    u8 day;
    /// Month [1, 12]
    u8 month;
    /// Year
    i32 year;
} datetime;

/// File stats
typedef struct {
    /// Size of file in bytes
    u64 size;
    /// File flags
    file_flags flags;
    /// Last time of modification
    datetime modify_time;
} file_stats;

/// Thread mutex
typedef struct _mutex mutex;

/// Thread pool
typedef struct _thread_pool thread_pool;

/// Function for thread to run
typedef void (thread_func)(void*);
/**
 * @brief Task for thread to run
 *
 * Function will do `func(arg)`
 */
typedef struct {
    /// Function to run
    thread_func* func;
    /// Function arg
    void* arg;
} thread_task;

/**
 * @brief Initialize time system
 *
 * This is because of the windows API, see `os_windows.c` for more
 */
void time_init(void);

/// Returns the local time
datetime now_localtime(void);
/// Returns time in microseconds
u64 now_usec(void);
/// Sleeps for `t` milliseconds
void sleep_msec(u32 t);

/**
 * @brief Reads entire file and returns it as a string8
 */
string8 file_read(mg_arena* arena, string8 path);
/**
 * @brief Writes all strings in list to file
 *
 * @return true if write was successful, false otherwise
 */
b32 file_write(string8 path, string8_list str_list);
/**
 * @brief Gets stats of file
 */
file_stats file_get_stats(string8 path);

/**
 * @brief Retrieves entropy from the OS
 *
 * @param data Where entropy is written to
 * @param size Number of bytes to retrieve
 */
void get_entropy(void* data, u64 size);

/**
 * @brief Creates a `mutex`
 *
 * @return Pointer to mutex on success, NULL on failure
 */
mutex* mutex_create(mg_arena* arena);
/// Destroys the mutex
void mutex_destroy(mutex* mutex);
/// Locks the mutex, returns true on success
b32 mutex_lock(mutex* mutex);
/// Unlocks the mutex, returns true on success
b32 mutex_unlock(mutex* mutex);

/**
 * @brief Creates a `thread_pool`
 *
 * @param arena Arena to allocate thread pool on
 * @param num_threads Number of threads in pool.
 *  It should not be much higher than the number of threads on your computer
 * @param max_tasks Maximum number of tasks that can be active at one time
 *
 * @return Pointer to thread pool on success, NULL on failure
 */
thread_pool* thread_pool_create(mg_arena* arena, u32 num_threads, u32 max_tasks);
/// Destroys the thread pool
void thread_pool_destroy(thread_pool* tp);
/**
 * @brief Adds task to the thread pool's task queue
 *
 * @return true if task was added successfully
 */
b32 thread_pool_add_task(thread_pool* tp, thread_task task);
/**
 * @brief Waits until all thread tasks are finished
 *
 * @return ture if waiting was successful
 */
b32 thread_pool_wait(thread_pool* tp);

#endif // OS_H