#pragma once
#include <cutlass/subbyte_reference.h>

namespace cutlass {

template <typename Element_,  /// CUTLASS numeric element type.
          typename Storage_   /// Underlying storage type. Must be able to hold
                              /// an integer
          >
class MySubbyteReference {
 public:
  using Element = Element_;
  using Storage = Storage_;
  using StoragePointer = Storage *;

  static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value,
                "Size of Element must not be greater than Storage.");

  static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value),
                "Storage must be divisible by Element");

  constexpr static int const kElementsPerVector =
      sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

 private:
  ///! Number of elements per storage vector

  ///! Bit mask
  Storage const kMask =
      ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value)
           ? (Storage(1) << sizeof_bits<Element>::value) - Storage(1)
           : ~Storage(0));

 private:
  /// Pointer to array containing element
  StoragePointer ptr_;

  /// Offset (in units of elements) from pointer.
  ///
  /// Invariant: must always be in range [0, kElementsPerVector)
  int offset_;

 public:
  CUTLASS_HOST_DEVICE
  MySubbyteReference() : ptr_(nullptr), offset_(0) {}

  /// Constructor
  CUTLASS_HOST_DEVICE
  MySubbyteReference(Element *ptr,   /// pointer to memory
                     int64_t offset  /// logical offset in units of Element
                     )
      : ptr_(reinterpret_cast<StoragePointer>(ptr)), offset_(0) {
    int64_t offset_in_vectors = offset / kElementsPerVector;
    int64_t offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = int(offset_in_elements);
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  MySubbyteReference(Element *ptr = nullptr) : MySubbyteReference(ptr, 0) {}

  /// Gets storage pointer
  CUTLASS_HOST_DEVICE
  StoragePointer storage_pointer() const { return ptr_; }

  /// Gets storage pointer
  CUTLASS_HOST_DEVICE
  Element *operator&() const { return reinterpret_cast<Element *>(ptr_); }

  /// Gets element offset within storage vector
  CUTLASS_HOST_DEVICE
  int element_offset() const { return offset_; }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  Element get() const {
    Storage item =
        Storage((*ptr_ >> (offset_ * sizeof_bits<Element>::value)) & kMask);
    return reinterpret_cast<Element const &>(item);
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  MySubbyteReference &set(Element const &x) {
    Storage item = (reinterpret_cast<Storage const &>(x) & kMask);
    Storage kUpdateMask =
        Storage(~(kMask << (offset_ * cutlass::sizeof_bits<Element>::value)));
    Storage new_bits =
        Storage(item << (offset_ * cutlass::sizeof_bits<Element>::value));

    Storage original = (*ptr_);
    Storage updated = Storage((original & kUpdateMask) | new_bits);
    *ptr_ = updated;

    return *this;
  }

  ////

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  operator Element() const { return get(); }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  MySubbyteReference &operator=(Element const &x) { return set(x); }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  MySubbyteReference &operator=(MySubbyteReference const &x) {
    return set(x.get());
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  MySubbyteReference &operator=(
      ConstSubbyteReference<Element, Storage> const &x) {
    return set(x.get());
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  MySubbyteReference &operator+=(int offset) {
    offset += offset_;

    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  MySubbyteReference &operator+=(long long offset) {
    offset += offset_;

    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  MySubbyteReference &operator-=(int offset) {
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  MySubbyteReference &operator-=(long long offset) {
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    return *this;
  }

  /// Returns a reference to an element with a given offset from the current
  /// reference
  CUTLASS_HOST_DEVICE
  MySubbyteReference operator+(int offset) const {
    MySubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current
  /// reference
  CUTLASS_HOST_DEVICE
  MySubbyteReference operator+(long long offset) const {
    MySubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current
  /// reference
  CUTLASS_HOST_DEVICE
  MySubbyteReference operator-(int offset) const {
    MySubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current
  /// reference
  CUTLASS_HOST_DEVICE
  MySubbyteReference operator-=(long long offset) const {
    MySubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Computes the difference in elements between references
  CUTLASS_HOST_DEVICE
  ptrdiff_t operator-(MySubbyteReference ref) const {
    return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
  }

  /// Explicit cast to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const { return int(get()); }

  /// Explicit cast to signed 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator int64_t() const { return int64_t(get()); }

  /// Explicit cast to unsigned 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator uint64_t() const { return uint64_t(get()); }

  /// Explicit cast to float
  CUTLASS_HOST_DEVICE
  explicit operator float() const { return float(get()); }

  /// Explicit cast to double
  CUTLASS_HOST_DEVICE
  explicit operator double() const { return double(get()); }
};

}  // namespace cutlass

using Int4Subbyte = cutlass::MySubbyteReference<cutlass::int4b_t, uint8_t>;
using Int4Storage = Int4Subbyte::Storage;
constexpr const uint32_t kElementsPerVector =
    cutlass::sizeof_bits<Int4Storage>::value /
    cutlass::sizeof_bits<cutlass::int4b_t>::value;
