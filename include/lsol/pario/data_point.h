/*********************************************************************************
*     File Name           :     data_point.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-29 11:16]
*     Last Modified       :     [2016-02-18 23:55]
*     Description         :     Data Point Structure
**********************************************************************************/

#ifndef LSOL_PARIO_DATA_POINT_H__
#define LSOL_PARIO_DATA_POINT_H__

#include <lsol/util/types.h>
#include <lsol/pario/array1d.h>

namespace lsol {
namespace pario {

/**
 * \brief  one label, one feature index vector, and one feature value vector
 */
class LSOL_EXPORTS DataPoint {
 public:
  /// \brief  Create an empty data point
  DataPoint();

  ~DataPoint() {}

  /// \brief  Clone the current point to destination point
  ///
  /// \param dst_pt destination point
  void Clone(DataPoint &dst_pt) const;

  /// \brief  Clone a new point from this
  ///
  /// \return new data point
  DataPoint Clone() const;

 public:
  /// \brief  add new feature into the data point, mostly used when loading
  // data
  ///
  /// \param index index of the feature
  /// \param feat value of the feature
  void AddNewFeat(index_t index, real_t feat);

  /// \brief  clear the label, indexes, and features
  void Clear();

  /// \brief  Check if the indexes are sorted from small to large
  ///
  /// \return true of sorted, false otherwise
  bool IsSorted() const;

  /// \brief  Sort the features so that indexes are from small to large
  void Sort();

 public:
  const Array1d<index_t> &indexes() const { return this->indexes_; }
  Array1d<index_t> &indexes() { return this->indexes_; }

  const Array1d<real_t> &features() const { return this->features_; }
  Array1d<real_t> &features() { return this->features_; }

  const index_t indexes(size_t index) const { return this->indexes_[index]; }

  const real_t features(index_t index) const { return this->features_[index]; }
  real_t &features(index_t index) { return this->features_[index]; }

  label_t label() const { return this->label_; }
  void set_label(label_t label) { this->label_ = label; }
  index_t dim() const {
    return this->indexes_.size() > 0 ? this->indexes_.last() : 0;
  }
  size_t size() const { return this->indexes_.size(); }

 protected:
  Array1d<index_t> indexes_;
  Array1d<real_t> features_;
  label_t label_;
};  // class DataPoint

}  // namespace pario
}  // namespace lsol

#endif
