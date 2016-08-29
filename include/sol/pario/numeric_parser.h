/*********************************************************************************
*     File Name           :     numeric_parser.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-11 22:52]
*     Last Modified       :     [2015-11-12 18:01]
*     Description         :     Numeric Parser
**********************************************************************************/
#ifndef SOL_PARIO_NUMERIC_PARSER_H__
#define SOL_PARIO_NUMERIC_PARSER_H__

#include <cmath>

namespace sol {
namespace pario {

class SOL_EXPORTS NumericParser {
 public:
  static inline bool is_space(char* p) {
    return (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r');
  }

  static inline char* strip_line(char* p) {
    while (is_space(p) == true) p++;
    return p;
  }

  // The following function is a home made strtoi
  static inline int ParseInt(char* p, char*& end) {
    end = p;
    p = strip_line(p);

    // no input, return 0, end == p
    if (*p == '\0') return 0;

    int s = 1;
    if (*p == '+')
      p++;
    else if (*p == '-') {
      s = -1;
      p++;
    }
    int acc = 0;
    while (*p >= '0' && *p <= '9') acc = acc * 10 + *p++ - '0';

    int num_dec = 0;
    if (*p == '.') {
      p++;
      while (*p >= '0' && *p <= '9') {
        acc = acc * 10 + *p++ - '0';
        num_dec++;
      }
    }
    int exp_acc = 0;
    if (*p == 'e' || *p == 'E') {
      p++;
      if (*p == '+') p++;
      while (*p >= '0' && *p <= '9') exp_acc = exp_acc * 10 + *p++ - '0';
    }
    if (exp_acc < num_dec)
      return 0;
    else if (exp_acc > 0)
      acc *= (int)(powf(10.f, (float)(exp_acc - num_dec)));

    end = strip_line(p);
    return s * acc;
  }

  // The following function is a home made strtoi
  static inline unsigned int ParseUint(char* p, char*& end) {
    end = p;
    p = strip_line(p);

    if (*p == '\0') return 0;
    unsigned int acc = 0;
    while (*p >= '0' && *p <= '9') acc = acc * 10 + *p++ - '0';

    int num_dec = 0;
    if (*p == '.') {
      p++;
      while (*p >= '0' && *p <= '9') {
        acc = acc * 10 + *p++ - '0';
        num_dec++;
      }
    }
    int exp_acc = 0;
    if (*p == 'e' || *p == 'E') {
      p++;
      if (*p == '+') p++;
      while (*p >= '0' && *p <= '9') exp_acc = exp_acc * 10 + *p++ - '0';
    }
    if (exp_acc < num_dec)
      return 0;
    else if (exp_acc > 0)
      acc *= (unsigned int)(powf(10.f, (float)(exp_acc - num_dec)));
    end = strip_line(p);
    return acc;
  }

  // The following function is a home made strtof. The
  // differences are :
  //  - much faster (around 50% but depends on the string to parse)
  //  - less error control, but utilised inside a very strict parser
  //    in charge of error detection.
  static inline float ParseFloat(char* p, char*& end) {
    end = p;
    p = strip_line(p);

    if (*p == '\0') return 0;

    int s = 1;
    if (*p == '+') p++;
    if (*p == '-') {
      s = -1;
      p++;
    }

    int acc = 0;
    while (*p >= '0' && *p <= '9') acc = acc * 10 + *p++ - '0';

    int num_dec = 0;
    if (*p == '.') {
      p++;
      while (*p >= '0' && *p <= '9' && num_dec != 7) {
        acc = acc * 10 + *p++ - '0';
        num_dec++;
      }
      while (*p >= '0' && *p <= '9') p++;
    }

    int exp_acc = 0;
    if (*p == 'e' || *p == 'E') {
      p++;
      int exp_s = 1;
      if (*p == '+') p++;
      if (*p == '-') {
        exp_s = -1;
        p++;
      }
      while (*p >= '0' && *p <= '9') exp_acc = exp_acc * 10 + *p++ - '0';
      exp_acc *= exp_s;
    }
    exp_acc -= num_dec;
    end = strip_line(p);
    if (exp_acc == 0) {
      return float(s * acc);
    } else {
      return s * acc * powf(10.f, (float)(exp_acc));
    }
  }

};  // class NumericParser

}  // namespace pario
}  // namespace sol

#endif
