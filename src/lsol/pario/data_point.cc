/*********************************************************************************
*     File Name           :     data_point.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-29 13:12]
*     Last Modified       :     [2015-11-13 21:01]
*     Description         :     Data Point Structure
**********************************************************************************/

#include "lsol/pario/data_point.h"

#include <utility>

namespace lsol {
namespace pario {

DataPoint::DataPoint() : label_(0) {}

void DataPoint::Clone(DataPoint &dst_pt) const {
    dst_pt.label_ = this->label_;
    dst_pt.indexes_.Reserve(this->indexes_.size());
    dst_pt.indexes_.Resize(this->indexes_.size());
    memcpy(dst_pt.indexes_.begin(), this->indexes_.begin(),
           this->indexes_.size() * sizeof(index_t));
    dst_pt.features_.Reserve(this->features_.size());
    dst_pt.features_.Resize(this->features_.size());
    memcpy(dst_pt.features_.begin(), this->features_.begin(),
           this->features_.size() * sizeof(real_t));
}

DataPoint DataPoint::Clone() const {
    DataPoint dst_pt;
    this->Clone(dst_pt);
    return dst_pt;
}

void DataPoint::AddNewFeat(index_t index, real_t feat) {
    this->indexes_.Push(index);
    this->features_.Push(feat);
}

void DataPoint::Clear() {
    this->indexes_.Clear();
    this->features_.Clear();
    this->label_ = 0;
}

bool DataPoint::IsSorted() const {
    for (auto iter = this->indexes_.begin() + 1; iter < this->indexes_.end();
         ++iter) {
        if (*iter <= *(iter - 1)) return false;
    }
    return true;
}

template <typename T1, typename T2>
void QuickSort(T1 *a, T2 *b, size_t low, size_t high) {  // from small to great
    size_t i = low;
    size_t j = high;
    T1 temp = a[low];  // select the first element as the indicator
    T2 temp_ind = b[low];

    while (i < j) {
        while ((i < j) && (temp < a[j])) {  // scan right side
            j--;
        }
        if (i < j) {
            a[i] = a[j];
            b[i] = b[j];
            i++;
        }

        while (i < j && (a[i] < temp)) {  // scan left side
            i++;
        }
        if (i < j) {
            a[j] = a[i];
            b[j] = b[i];
            j--;
        }
    }
    a[i] = temp;
    b[i] = temp_ind;

    if (low < i) {
        QuickSort(a, b, low, i - 1);  // sort left subset
    }
    if (i < high) {
        QuickSort(a, b, j + 1, high);  // sort right subset
    }
}

void DataPoint::Sort() {
    if (this->IsSorted() == false) {
        QuickSort(this->indexes_.begin(), this->features_.begin(), 0,
                  this->indexes_.size() - 1);
    }
}

}  // namespace pario
}  // namespace lsol
