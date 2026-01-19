#include "../../include/os.h"
#include "../../include/err.h"
#include "../random_generators/prng.h"

#if defined(__APPLE__)


#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <pthread.h>
#include <sys/random.h>

void time_init(void) { }

datetime _tm_to_datetime(struct tm tm) {
    return (datetime){
        .sec = tm.tm_sec,
        .min = tm.tm_min,
        .hour = tm.tm_hour,
        .day = tm.tm_mday,
        .month = tm.tm_mon + 1,
        .year = tm.tm_year + 1900
    };
}

datetime now_localtime(void) {
    time_t t = time(NULL);
    struct tm* tm_ptr = localtime(&t);

    if (tm_ptr == NULL) {
        ERR(ERR_OS, "Failed to convert time to localtime");

        return (datetime){ 0 };
    }

    struct tm tm = *tm_ptr;

    return _tm_to_datetime(tm);
}

u64 now_usec(void) {
    struct timespec ts = { 0 };
    if (-1 == clock_gettime(CLOCK_MONOTONIC, &ts)) {
        ERR(ERR_OS, "Failed to get current time");
        return 0;
    }

    return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

void sleep_msec(u32 t) {
    usleep(t * 1000);
}

int _open_impl(string8 path, int flags, mode_t mode) {
    mga_temp scratch = mga_scratch_get(NULL, 0);
    
    u8* path_cstr = str8_to_cstr(scratch.arena, path);
    int fd = open((char*)path_cstr, flags, mode);

    mga_scratch_release(scratch);

    return fd;
}

string8 file_read(mg_arena* arena, string8 path) {
    int fd = _open_impl(path, O_RDONLY, 0);
    
    if (fd == -1) {
        ERR(ERR_IO, "Failed to open file for reading");

        return (string8){ 0 };
    }
    
    struct stat file_stat = { 0 };
    // This should not really error if the file was opened correctly
    fstat(fd, &file_stat);

    string8 out = { 0 };

    if (S_ISREG(file_stat.st_mode)) {
        out.size = file_stat.st_size;
        out.str = MGA_PUSH_ZERO_ARRAY(arena, u8, (u64)file_stat.st_size);

        if (read(fd, out.str, file_stat.st_size) == -1) {
            ERR(ERR_IO, "Failed to read file");
            
            close(fd);
            
            return (string8){ 0 };
        }
    } else {
        ERR(ERR_IO, "Failed to read file, file is not regular (most likely a directory)");
    }

    if (-1 == close(fd)) {
        ERR(ERR_IO, "Failed to close file");
    }

    return out;
}

b32 file_write(string8 path, string8_list str_list) {
    int fd = _open_impl(path, O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);

    if (fd == -1) {
        ERR(ERR_IO, "Failed to open file for writing");

        return false;
    }

    b32 out = true;
    
    for (string8_node* node = str_list.first; node != NULL; node = node->next) {
        ssize_t written = write(fd, node->str.str, node->str.size);

        if (written == -1) {
            ERR(ERR_IO, "Failed to write to file");

            out = false;
            break;
        }
    }
        
    if (-1 == close(fd)) {
        ERR(ERR_IO, "Failed to close file");
    }

    return out;
}

file_flags _file_flags(mode_t mode) {
    file_flags flags = { 0 };

    if (S_ISDIR(mode)) {
        flags |= FILE_IS_DIR;
    }

    return flags;
}

file_stats file_get_stats(string8 path) {
    mga_temp scratch = mga_scratch_get(NULL, 0);
    
    u8* path_cstr = str8_to_cstr(scratch.arena, path);
    
    struct stat file_stat = { 0 };
    
    int ret = stat((char*)path_cstr, &file_stat);
    
    mga_scratch_release(scratch);
    
    if (ret == -1) {
        ERR(ERR_IO, "Failed to get stats for file");

        return (file_stats){ 0 };
    } 

    file_stats stats = { 0 };
    
    // macOS uses st_mtimespec instead of st_mtim
    time_t modify_time = file_stat.st_mtimespec.tv_sec;
    struct tm* tm_ptr = localtime(&modify_time);
    struct tm tm = { 0 };

    // This error is recoverable
    if (tm_ptr == NULL) {
        ERR(ERR_OS, "Failed to convert time_t to localtime");
    } else {
        tm = *tm_ptr;
    }

    stats.size = file_stat.st_size;
    stats.flags = _file_flags(file_stat.st_mode);
    stats.modify_time = _tm_to_datetime(tm);

    return stats;
}

void get_entropy(void* data, u64 size) {
    // macOS has getentropy() available since macOS 10.12
    // For compatibility with older versions, use arc4random_buf as fallback
    #if __MAC_OS_X_VERSION_MIN_REQUIRED >= 101200
        if (-1 == getentropy(data, size)) {
            ERR(ERR_OS, "Failed to get entropy from system");
        }
    #else
        // Fallback for older macOS versions
        arc4random_buf(data, size);
    #endif
}

typedef struct _mutex {
    pthread_mutex_t mutex;
} mutex;

// TODO: check mutex for NULL
mutex* mutex_create(mg_arena* arena) {
    pthread_mutex_t pmutex = PTHREAD_MUTEX_INITIALIZER;

    if (0 != pthread_mutex_init(&pmutex, NULL)) {
        ERR(ERR_THREADING, "Failed to create mutex");

        return NULL;
    }

    mutex* result = MGA_PUSH_ZERO_STRUCT(arena, mutex);
    result->mutex = pmutex;

    return result;
}

void mutex_destroy(mutex* mutex) {
    if (0 != pthread_mutex_destroy(&mutex->mutex)) {
        // Is there something else to do here?
        ERR(ERR_THREADING, "Failed to delete mutex");
    }
}

b32 mutex_lock(mutex* mutex) {
    if (0 != pthread_mutex_lock(&mutex->mutex)) {
        ERR(ERR_THREADING, "Failed to lock mutex");

        return false;
    }
    return true;
}

b32 mutex_unlock(mutex* mutex) {
    if (0 != pthread_mutex_unlock(&mutex->mutex)) {
        ERR(ERR_THREADING, "Failed to unlock mutex");

        return false;
    }

    return true;
}

typedef struct _thread_pool {
    u32 num_threads;
    pthread_t* threads;
    
    b32 stop;

    u32 max_tasks;
    u32 num_tasks;
    thread_task* task_queue;

    pthread_mutex_t mutex;
    pthread_cond_t queue_cond_var;

    u32 num_active;
    pthread_cond_t active_cond_var;
} thread_pool;

static void* macos_thread_start(void* arg) {
    thread_pool* tp = (thread_pool*)arg;
    thread_task task = { 0 };

    // Init prng
    u64 seeds[2] = { 0 };
    get_entropy(seeds, sizeof(seeds));
    prng_seed(seeds[0], seeds[1]);

    while (true) {
        pthread_mutex_lock(&tp->mutex);

        while (tp->num_tasks == 0 && !tp->stop) {
            pthread_cond_wait(&tp->queue_cond_var, &tp->mutex);
        }

        if (tp->stop) {
            break;
        }

        tp->num_active++;
        task = tp->task_queue[0];
        for (u32 i = 0; i < tp->num_tasks - 1; i++) {
            tp->task_queue[i] = tp->task_queue[i + 1];
        }
        tp->num_tasks--;

        pthread_mutex_unlock(&tp->mutex);

        task.func(task.arg);

        pthread_mutex_lock(&tp->mutex);

        tp->num_active--;
        if (tp->num_active == 0) {
            pthread_cond_signal(&tp->active_cond_var);
        }

        pthread_mutex_unlock(&tp->mutex);
    }

    tp->num_threads--;
    pthread_cond_signal(&tp->active_cond_var);
    pthread_mutex_unlock(&tp->mutex);

    return NULL;
}

thread_pool* thread_pool_create(mg_arena* arena, u32 num_threads, u32 max_tasks) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    thread_pool* tp = MGA_PUSH_ZERO_STRUCT(arena, thread_pool);

    tp->max_tasks = MAX(num_threads, max_tasks);
    tp->task_queue = MGA_PUSH_ZERO_ARRAY(arena, thread_task, max_tasks);

    i32 ret = 0;
    ret |= pthread_mutex_init(&tp->mutex, NULL);
    ret |= pthread_cond_init(&tp->queue_cond_var, NULL);
    ret |= pthread_cond_init(&tp->active_cond_var, NULL);

    if (ret != 0) {
        ERR(ERR_THREADING, "Failed to init thread pool mutex and/or cond vars");

        mga_temp_end(maybe_temp);

        return NULL;
    }

    tp->num_threads = num_threads;
    tp->threads = MGA_PUSH_ZERO_ARRAY(arena, pthread_t, num_threads);
    for (u32 i = 0; i < num_threads; i++) {
        ret = 0;

        ret |= pthread_create(&tp->threads[i], NULL, macos_thread_start, tp);
        ret |= pthread_detach(tp->threads[i]);

        if (ret != 0) {
            ERR(ERR_THREADING, "Failed to create or detach threads in thread pool");

            for (u32 j = 0; j < i; j++) {
                pthread_cancel(tp->threads[j]);
            }

            pthread_cond_destroy(&tp->active_cond_var);
            pthread_mutex_destroy(&tp->mutex);
            pthread_cond_destroy(&tp->queue_cond_var);

            mga_temp_end(maybe_temp);

            return NULL;
        }
    }

    return tp;
}

void thread_pool_destroy(thread_pool* tp) {
    if (tp == NULL) {
        ERR(ERR_INVALID_INPUT, "Unable to destroy NULL thread pool");

        return;
    }

    if (0 != pthread_mutex_lock(&tp->mutex)) {
        ERR(ERR_THREADING, "Unable to destroy threadpool: cannot lock mutex");

        return;
    }

    tp->num_tasks = 0;

    tp->stop = true;
    pthread_cond_broadcast(&tp->queue_cond_var);

    pthread_mutex_unlock(&tp->mutex);

    thread_pool_wait(tp);

    // Everything here should destroy correctly if the tp was initialized correctly 

    for (u32 i = 0; i < tp->num_threads; i++) {
        pthread_cancel(tp->threads[i]);
    }

    pthread_mutex_destroy(&tp->mutex);
    pthread_cond_destroy(&tp->queue_cond_var);
    pthread_cond_destroy(&tp->active_cond_var);
}

b32 thread_pool_add_task(thread_pool* tp, thread_task task) {
    if (tp == NULL) {
        ERR(ERR_INVALID_INPUT, "Unable to add task to NULL thread pool");

        return false;
    }

    if (0 != pthread_mutex_lock(&tp->mutex)) {
        ERR(ERR_THREADING, "Failed to lock thread pool mutex, unable to add task");

        return false;
    }

    if ((u64)tp->num_tasks + 1 > (u64)tp->max_tasks) {
        pthread_mutex_unlock(&tp->mutex);

        ERR(ERR_THREADING, "Thread pool exceeded max tasks");

        return false;
    }

    tp->task_queue[tp->num_tasks++] = task;

    pthread_mutex_unlock(&tp->mutex);

    pthread_cond_signal(&tp->queue_cond_var);

    return true;
}

b32 thread_pool_wait(thread_pool* tp) {
    if (tp == NULL) {
        ERR(ERR_INVALID_INPUT, "Unable to wait for NULL thread pool");

        return false;
    }

    if (0 != pthread_mutex_lock(&tp->mutex)) {
        ERR(ERR_THREADING, "Cannot wait for thread pool: unable to lock mutex");

        return false;
    }

    while (true) {
        //if (tp->num_active != 0 || tp->num_tasks != 0) {
        if ((!tp->stop && (tp->num_active != 0 || tp->num_tasks != 0)) || (tp->stop && tp->num_threads != 0)) {
            pthread_cond_wait(&tp->active_cond_var, &tp->mutex);
        } else {
            break;
        }
    }

    pthread_mutex_unlock(&tp->mutex);

    return true;
}

#endif