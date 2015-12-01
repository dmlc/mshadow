/*!
 *  Copyright (c) 2015 by Contributors
 * \file logging.h
 * \brief defines logging macros of dmlc
 *  allows use of GLOG, fall back to internal
 *  implementation when disabled
 */
#ifndef MSHADOW_LOGGING_H_
#define MSHADOW_LOGGING_H_
#ifndef DMLC_LOGGING_H_
#define DMLC_LOGGING_H_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
#include "./base.h"

namespace dmlc {
/*! \brief taken from DMLC directly */

/*!
 * \brief exception class that will be thrown by
 *  default logger if DMLC_LOG_FATAL_THROW == 1
 */
struct Error : public std::runtime_error {
  /*!
   * \brief constructor
   * \param s the error message
   */
  explicit Error(const std::string &s) : std::runtime_error(s) {}
};
}  // namespace dmlc

#if defined(_MSC_VER) && _MSC_VER < 1900
#define noexcept(a)
#endif

#if DMLC_USE_CXX11
#define DMLC_THROW_EXCEPTION noexcept(false)
#else
#define DMLC_THROW_EXCEPTION
#endif

#if DMLC_USE_GLOG
#include <glog/logging.h>

namespace dmlc {
/*! \brief taken from DMLC directly */
inline void InitLogging(const char* argv0) {
  google::InitGoogleLogging(argv0);
}
}  // namespace dmlc

#else
// use a light version of glog
#include <assert.h>
#include <iostream>
#include <sstream>
#include <ctime>

#if defined(_MSC_VER)
#pragma warning(disable : 4722)
#endif

namespace dmlc {
inline void InitLogging(const char* argv0) {
  // DO NOTHING
}

// Always-on checking
#define MSHADOW_CHECK(x)                                           \
  if (!(x))                                                \
    dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "Check "  \
      "failed: " #x << ' '
#define MSHADOW_CHECK_LT(x, y) MSHADOW_CHECK((x) < (y))
#define MSHADOW_CHECK_GT(x, y) MSHADOW_CHECK((x) > (y))
#define MSHADOW_CHECK_LE(x, y) MSHADOW_CHECK((x) <= (y))
#define MSHADOW_CHECK_GE(x, y) MSHADOW_CHECK((x) >= (y))
#define MSHADOW_CHECK_EQ(x, y) MSHADOW_CHECK((x) == (y))
#define MSHADOW_CHECK_NE(x, y) MSHADOW_CHECK((x) != (y))
#define MSHADOW_CHECK_NOTNULL(x) \
  ((x) == NULL ? dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "Check  notnull: "  #x << ' ', (x) : (x)) // NOLINT(*)
// Debug-only checking.
#ifdef NDEBUG
#define MSHADOW_DCHECK(x) \
  while (false) MSHADOW_CHECK(x)
#define MSHADOW_DCHECK_LT(x, y) \
  while (false) MSHADOW_CHECK((x) < (y))
#define MSHADOW_DCHECK_GT(x, y) \
  while (false) MSHADOW_CHECK((x) > (y))
#define MSHADOW_DCHECK_LE(x, y) \
  while (false) MSHADOW_CHECK((x) <= (y))
#define MSHADOW_DCHECK_GE(x, y) \
  while (false) MSHADOW_CHECK((x) >= (y))
#define MSHADOW_DCHECK_EQ(x, y) \
  while (false) MSHADOW_CHECK((x) == (y))
#define MSHADOW_DCHECK_NE(x, y) \
  while (false) MSHADOW_CHECK((x) != (y))
#else
#define MSHADOW_DCHECK(x) MSHADOW_CHECK(x)
#define MSHADOW_DCHECK_LT(x, y) MSHADOW_CHECK((x) < (y))
#define MSHADOW_DCHECK_GT(x, y) MSHADOW_CHECK((x) > (y))
#define MSHADOW_DCHECK_LE(x, y) MSHADOW_CHECK((x) <= (y))
#define MSHADOW_DCHECK_GE(x, y) MSHADOW_CHECK((x) >= (y))
#define MSHADOW_DCHECK_EQ(x, y) MSHADOW_CHECK((x) == (y))
#define MSHADOW_DCHECK_NE(x, y) MSHADOW_CHECK((x) != (y))
#endif  // NDEBUG

#define MSHADOW_LOG_INFO dmlc::LogMessage(__FILE__, __LINE__)
#define MSHADOW_LOG_ERROR MSHADOW_LOG_INFO
#define MSHADOW_LOG_WARNING MSHADOW_LOG_INFO
#define MSHADOW_LOG_FATAL dmlc::LogMessageFatal(__FILE__, __LINE__)
#define MSHADOW_LOG_QFATAL MSHADOW_LOG_FATAL

// Poor man version of VLOG
#define MSHADOW_VLOG(x) MSHADOW_LOG_INFO.stream()

#define MSHADOW_LOG(severity) MSHADOW_LOG_##severity.stream()
#define MSHADOW_LG MSHADOW_LOG_INFO.stream()
#define MSHADOW_LOG_IF(severity, condition) \
  !(condition) ? (void)0 : dmlc::LogMessageVoidify() & MSHADOW_LOG(severity)

#ifdef NDEBUG
#define MSHADOW_LOG_DFATAL MSHADOW_LOG_ERROR
#define MSHADOW_DFATAL ERROR
#define MSHADOW_DLOG(severity) true ? (void)0 : dmlc::LogMessageVoidify() & MSHADOW_LOG(severity)
#define MSHADOW_DLOG_IF(severity, condition) \
  (true || !(condition)) ? (void)0 : dmlc::LogMessageVoidify() & MSHADOW_LOG(severity)
#else
#define MSHADOW_LOG_DFATAL MSHADOW_LOG_FATAL
#define MSHADOW_DFATAL FATAL
#define MSHADOW_DLOG(severity) MSHADOW_LOG(severity)
#define MSHADOW_DLOG_IF(severity, condition) MSHADOW_LOG_IF(severity, condition)
#endif

// Poor man version of LOG_EVERY_N
#define MSHADOW_LOG_EVERY_N(severity, n) MSHADOW_LOG(severity)

class DateLogger {
 public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif
  }
  const char* HumanDate() {
#if defined(_MSC_VER)
    _strtime_s(buffer_, sizeof(buffer_));
#else
    time_t time_value = time(NULL);
    struct tm now;
    localtime_r(&time_value, &now);
    snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d", now.tm_hour,
             now.tm_min, now.tm_sec);
#endif
    return buffer_;
  }
 private:
  char buffer_[9];
};

class LogMessage {
 public:
  LogMessage(const char* file, int line)
      :
#ifdef __ANDROID__
        log_stream_(std::cout)
#else
        log_stream_(std::cerr)
#endif
  {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  ~LogMessage() { log_stream_ << "\n"; }
  std::ostream& stream() { return log_stream_; }

 protected:
  std::ostream& log_stream_;

 private:
  DateLogger pretty_date_;
  LogMessage(const LogMessage&);
  void operator=(const LogMessage&);
};

#if DMLC_LOG_FATAL_THROW == 0
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) : LogMessage(file, line) {}
  ~LogMessageFatal() {
    log_stream_ << "\n";
    abort();
  }

 private:
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#else
class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  std::ostringstream &stream() { return log_stream_; }
  ~LogMessageFatal() DMLC_THROW_EXCEPTION {
    // throwing out of destructor is evil
    // hopefully we can do it here
    throw Error(log_stream_.str());
  }

 private:
  std::ostringstream log_stream_;
  DateLogger pretty_date_;
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#endif

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
  void operator&(std::ostream&) {}
};

}  // namespace dmlc

#endif
#endif  // DMLC_LOGGING_H_
#endif  // MSHADOW_LOGGING_H_

