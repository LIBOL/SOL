/*********************************************************************************
*     File Name           :     data_point.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-29 11:16]
*     Last Modified       :     [2016-02-18 23:55]
*     Description         :     Data Point Structure
**********************************************************************************/

#ifndef SOL_PARIO_DATA_POINT_H__
#define SOL_PARIO_DATA_POINT_H__

#include <sol/util/types.h>
#include <sol/math/sparse_vector.h>

namespace sol {
namespace pario {

/**
 * \brief  one label, one feature index vector, and one feature value vector
 */
class SOL_EXPORTS DataPoint {
 public:
  /// \brief  Create an empty data point
  DataPoint();

  ~DataPoint() {}

  /// \brief  Clone the current point to destination point
  ///
  /// \param dst_pt destination point
  void Clone(DataPoint& dst_pt) const;

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

  inline void Reserve(size_t new_size) { this->data_.reserve(new_size); }
  inline void Resize(size_t new_size) { this->data_.resize(new_size); }
  /// \brief  clear the label, indexes, and features
  void Clear();

  /// \brief  Check if the indexes are sorted from small to large
  ///
  /// \return true of sorted, false otherwise
  bool IsSorted() const;

  /// \brief  Sort the features so that indexes are from small to large
  void Sort();

 public:
  inline const math::SVector<real_t>& data() const { return data_; }
  inline math::SVector<real_t>& data() { return data_; }

  inline const math::Vector<index_t>& indexes() const {
    return this->data_.indexes();
  }
  inline math::Vector<index_t>& indexes() { return this->data_.indexes(); }

  inline const math::Vector<real_t>& features() const {
    return this->data_.values();
  }
  inline math::Vector<real_t>& features() { return this->data_.values(); }

  inline index_t index(size_t index) const { return this->data_.index(index); }
  inline index_t& index(size_t index) { return this->data_.index(index); }

  inline real_t feature(size_t index) const { return this->data_.value(index); }
  inline real_t& feature(size_t index) { return this->data_.value(index); }

  inline label_t label() const { return this->label_; }
  inline void set_label(label_t label) { this->label_ = label; }
  index_t dim() const { return this->data_.dim(); }
  inline size_t size() const { return this->data_.size(); }

 protected:
  math::SVector<real_t> data_;
  label_t label_;
};  // class DataPoint

}  // namespace pario
}  // namespace sol

#endif
