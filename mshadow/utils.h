/*!
 *  Copyright (c) 2014 by Contributors
 * \file utils.h
 * \brief simple utils for error and checkings
 * \author Tianqi Chen
 */
#ifndef MSHADOW_UTILS_H_
#define MSHADOW_UTILS_H_
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <string>
#include <cstdarg>
#include <cstdlib>
namespace mshadow {
/*! \brief namespace for helper utils of the project */
namespace utils {
/*! \brief error message buffer length */
const int kPrintBuffer = 1 << 12;

#ifndef MSHADOW_CUSTOMIZE_ASSERT_
/*! 
 * \brief handling of Assert error, caused by in-apropriate input
 * \param msg error message 
 */
inline void HandleAssertError(const char *msg) {
  fprintf(stderr, "AssertError:%s\n", msg);
  exit(-1);
}
/*! 
 * \brief handling of Check error, caused by in-apropriate input
 * \param msg error message 
 */
inline void HandleCheckError(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  exit(-1);
}
#else
// include declarations, some one must implement this
void HandleAssertError(const char *msg);
void HandleCheckError(const char *msg);
void HandlePrint(const char *msg);
#endif

/*! \brief assert an condition is true, use this to handle debug information */
inline void Assert(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleAssertError(msg.c_str());
  }
}

/*!\brief same as assert, but this is intended to be used as message for user*/
inline void Check(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}

/*! \brief report error message, same as check */
inline void Error(const char *fmt, ...) {
  {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}
}  // namespace utils
}  // namespace mshadow
#endif  // MSHADOW_UTILS_H_
