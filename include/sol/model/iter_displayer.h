/*********************************************************************************
 *     File Name           :     iter_displayer.h
 *     Created By          :     yuewu
 *     Creation Date       :     [2017-05-08 22:56]
 *     Last Modified       :     [2017-05-09 16:28]
 *     Description         :     show iteration info related classes
 **********************************************************************************/
#ifndef SOL_MODEL_ITER_DISPLAYER_H__
#define SOL_MODEL_ITER_DISPLAYER_H__

#include <iomanip>
#include <iostream>

namespace sol {
namespace model {

/// \brief  C type to inspect iteration callback
///
/// \param user_context flexible place to handle iteration status
/// \param data_num number of data processed currently
/// \param iter_num number of iterations currently
/// \param update_num number of updates currently
/// \param err_rate training error rate currently
typedef void (*InspectIterateCallback)(void* user_context, long long data_num,
                                       long long iter_num, long long update_num,
                                       double err_rate);

class IterDisplayer {
 public:
  virtual ~IterDisplayer() {}

  virtual size_t next() { return size_t(-1); }
};

class ExpIterDisplayer : public IterDisplayer {
 public:
  ExpIterDisplayer(size_t base = 2)
      : next_show_time_(base), base_(base), show_step_(0) {}

  virtual inline size_t next() {
    ++show_step_;
    next_show_time_ = size_t(pow(double(this->base_), this->show_step_));
    return next_show_time_;
  }

 protected:
  size_t next_show_time_;
  size_t base_;
  size_t show_step_;
};

class StepIterDisplayer : public IterDisplayer {
 public:
  StepIterDisplayer(size_t step = 2) : next_show_time_(0), step_(step) {}

  virtual size_t next() {
    this->next_show_time_ += this->step_;
    return next_show_time_;
  }

 protected:
  size_t next_show_time_;
  size_t step_;
};

inline void DefaultIterateFunction(void* user_context, long long data_num,
                                   long long iter_num, long long update_num,
                                   double err_rate) {
  std::cout << data_num << "\t\t" << iter_num << "\t\t" << std::fixed
            << std::setprecision(6) << err_rate << "\t" << update_num << "\n";
}

}  // namespace model
}  // namespace sol
#endif
