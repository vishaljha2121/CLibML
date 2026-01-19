//
// Created by Vishal Jha on 16/01/26.
//

/**
 * @file err.h
 * @brief Error handling
 */

#ifndef ERR_H
#define ERR_H

#include "base_defs.h"
#include "str.h"

/**
 * @brief List of error codes
 *
 * To use, define a macro X, insert the definition, and then undef the macro X
 */
#define ERROR_XLIST \
X(ERR_NULL) \
X(ERR_GENERAL) \
X(ERR_INVALID_INPUT) \
X(ERR_ALLOC_SIZE) \
X(ERR_BAD_SHAPE) \
X(ERR_PARSE) \
X(ERR_IO) \
X(ERR_OS) \
X(ERR_THREADING) \
X(ERR_INVALID_ENUM) \
X(ERR_CREATE) \
X(ERR_MATH)

/**
 * @brief Error codes
 *
 * See `ERROR_XLIST` for full list
 */
typedef enum {
#define X(code) code,
    ERROR_XLIST
#undef X

    ERR_COUNT
} error_code;

/**
 * @brief Error code and message
 *
 * Used for error callbacks
 */
typedef struct {
    error_code code;
    string8 msg;
} error;

/// Error callback function
typedef void (error_callback)(error err);

/**
 * @brief Calls the global error callback with the error
 *
 * This is mainly meant for internal use
 */
void err(error err);

/**
 * @brief Calls `err`, converting the cstr to a `string8`
 *
 * @param err_code Error code
 * @param msg_cstr Error message as a c string
 */
#define ERR(err_code, msg_cstr) err((error){ .code=err_code, .msg=STR8(msg_cstr) })

/**
 * @brief Sets the error callback
 * 
 * This is called by any TurboSpork function that has an error. <br>
 * WARNING: This function will be called from multiple threads.
 * Anything done in this function needs to be threadsafe. <br>
 * The default error callback prints the information to `stderr`
 *
 * @param callback New error callback. Needs to be threadsafe
 */
void err_set_callback(error_callback* callback);

/// Converts a `error_code` to a `string8`. Do not modify the returned string
string8 err_to_str(error_code code);

/// Converst a `string8` to `error_code` 
error_code err_from_str(string8 str);

#endif // ERR_H