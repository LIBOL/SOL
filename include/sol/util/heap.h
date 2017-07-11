/*********************************************************************************
*     File Name           :     heap.h
*     Created By          :     yuewu
*     Description         :     Heap list to select topK elements
**********************************************************************************/

#ifndef SOL_UTIL_HEAP_H__
#define SOL_UTIL_HEAP_H__

#include <stdexcept>

#include <sol/math/vector.h>
#include <sol/util/types.h>

namespace sol {

/// \brief  Heap. Note that, for MinHeap, the K elements are structured as a
// MinHeap, while the other (N-K) elements should be smaller than the K
// elements. For MaxHeap, the other (N-K) elements should be larger than the K
// elements
///
/// \tparam comparator <++>
template <typename comparator>
class Heap {
 public:
  Heap() : N_(0), K_(0), values_(nullptr) {}

 public:
  /// \brief  initialize the heap with provided data
  ///
  /// \param N number of data in total
  /// \param K number of data to keep in heap
  /// \param values data values
  void Init(index_t N, index_t K, const real_t* values);

  /// \brief  initialize the heap with provided indexes and data
  ///
  /// \param N number of data in total
  /// \param K number of data to keep in heap
  /// \param indexes data indexes
  /// \param values data values
  void Init(index_t N, index_t K, const index_t* indexes, const real_t* values);

  /// \brief  build the heap so that r[0,K) is a heap
  void BuildHeap();

  /// \brief  heap sort
  void HeapSort();

  /// \brief  check if idx is outside of the heap and if it should be moved in
  // to the heap
  ///
  /// \param idx the index that needs to be updated
  ///
  /// \return element index that is moved out-of the heap
  index_t UpdateHeap(index_t idx);

  /// \brief  adjust the heap to satisfy heap properties
  ///
  /// \param s r[s+1,...,m] is heap, adjust the heap so that r[s,..,m] is heap
  /// \param m <++>
  void AdjustHeap(index_t s, index_t m);

 protected:
  /// \brief  release memory, for Init or destructor
  void Release();

 public:
  inline index_t K() const { return K_; }

  inline index_t get_pos(index_t idx) const { return this->id2pos_map_[idx]; }

  inline bool isTopK(index_t idx) const {
    return this->id2pos_map_[idx] < this->K_;
  }
  inline real_t value_by_pos(index_t pos) {
    return this->values_[this->pos2id_map_[pos]];
  }

  /// \brief  set the total number of data in heap
  ///
  /// \param N new data number
  /// \param values data values
  void set_N(index_t N, const real_t* values);

 protected:
  index_t N_;  // number of elements in total
  index_t K_;  // keep top K elemetns

  const real_t* values_;  // data array

  // sorted position of each data index
  math::Vector<index_t> id2pos_map_;

  // data index for sorted position
  math::Vector<index_t> pos2id_map_;
};

template <typename comparator>
void Heap<comparator>::Init(index_t N, index_t K, const real_t* values) {
  this->Release();

  this->N_ = N;
  this->id2pos_map_.resize(N);
  for (index_t i = 0; i != N; ++i) {
    this->id2pos_map_[i] = i;
  }

  this->K_ = K;
  this->pos2id_map_.resize(K);
  for (index_t i = 0; i != K; ++i) {
    this->pos2id_map_[i] = i;
  }

  this->values_ = values;
  this->BuildHeap();
}

template <typename comparator>
void Heap<comparator>::Init(index_t N, index_t K, const index_t* indexes,
                            const real_t* values) {
  this->Release();

  this->N_ = N;
  this->id2pos_map_.resize(N);
  for (index_t i = 0; i != N; ++i) {
    this->id2pos_map_[i] = K;
  }

  this->K_ = K;
  this->pos2id_map_.resize(K);
  for (index_t i = 0; i != K; ++i) {
    this->pos2id_map_[i] = indexes[i];
  }

  for (index_t i = 0; i != K; ++i) {
    this->id2pos_map_[this->pos2id_map_[i]] = i;
  }

  this->values_ = values;
  this->BuildHeap();
}

template <typename comparator>
void Heap<comparator>::BuildHeap() {
  index_t i = (this->K_ - 1) / 2 + 1;
  do {
    this->AdjustHeap(--i, this->K_ - 1);
  } while (i > 0);
}

template <typename comparator>
void Heap<comparator>::HeapSort() {
  this->BuildHeap();

  for (int i = this->K - 1; i > 0; --i) {
    // swap top and last
    index_t top_id = this->pos2id_map_[0];
    this->pos2id_map_[0] = this->pos2id_map_[i];
    this->id2pos_map_[this->pos2id_map_[i]] = 0;
    this->pos2id_map_[i] = top_id;
    this->id2pos_map_[top_id] = i;

    this->HeapAdjust(0, i - 1);
  }
}

template <typename comparator>
index_t Heap<comparator>::UpdateHeap(index_t idx) {
  index_t cur_pos = this->id2pos_map_[idx];
  if (cur_pos < this->K_) return invalid_index;

  real_t cur_val = this->values_[idx];

  real_t thresh = this->value_by_pos(0);
  if (comparator::map(thresh, cur_val)) {
    index_t ret_id = this->pos2id_map_[0];
    // swap with the top element of the heap
    this->id2pos_map_[ret_id] = this->K_;
    this->id2pos_map_[idx] = 0;
    this->pos2id_map_[0] = idx;
    this->AdjustHeap(0, this->K_ - 1);

    return ret_id;
  } else {
    return idx;
  }
}

template <typename comparator>
void Heap<comparator>::AdjustHeap(index_t s, index_t m) {
  index_t tgt_id = this->pos2id_map_[s];   // parent id
  real_t tgt_val = this->values_[tgt_id];  // parent value

  index_t parent_pos = s;
  index_t child_pos = 2 * parent_pos + 1;  // child id
  for (; child_pos <= m; child_pos = 2 * parent_pos + 1) {
    if (child_pos < m && comparator::map(this->value_by_pos(child_pos + 1),
                                         this->value_by_pos(child_pos)))
      ++child_pos;  // i+1 is the smaller(MinHeap)/larger(MaxHeap) child

    if (!comparator::map(this->value_by_pos(child_pos), tgt_val)) break;

    // set parent node to be child node
    index_t child_id = this->pos2id_map_[child_pos];
    this->pos2id_map_[parent_pos] = child_id;
    this->id2pos_map_[child_id] = parent_pos;
    parent_pos = child_pos;
  }

  // insert
  this->pos2id_map_[parent_pos] = tgt_id;
  this->id2pos_map_[tgt_id] = parent_pos;
}

template <typename comparator>
void Heap<comparator>::Release() {
  this->N_ = 0;
  this->K_ = 0;
  this->values_ = nullptr;
  this->id2pos_map_.resize(0);
  this->pos2id_map_.resize(0);
}

template <typename comparator>
void Heap<comparator>::set_N(index_t N, const real_t* values) {
  if (this->K_ == 0) return;
  if (N < this->N_) {
    throw std::invalid_argument(
        "data number in heap is not allowed to decrease!");
  }

  this->id2pos_map_.resize(N);
  // set the rest
  for (index_t i = this->N_; i < N; ++i) this->id2pos_map_[i] = i;
  this->N_ = N;

  this->values_ = values;
}

typedef Heap<math::expr::op::smaller> MinHeap;
typedef Heap<math::expr::op::larger> MaxHeap;
}  // namespace sol
#endif
