/*	
	This code comes from https://gist.github.com/mattyclarkson/5318077
	It is an implementation of the Murmur3 hashing algorithm:
		https://en.wikipedia.org/wiki/MurmurHash
		https://code.google.com/p/smhasher/

	Distributed under the MIT License.
*/

#pragma once


#include <type_traits>
#include <iostream>
#include <cassert>
#include <cstdint>


constexpr size_t _StringLengthLoop(const char * const str, const size_t size)
    noexcept {
  return (str[size] == '\0') ? size : _StringLengthLoop(str, size + 1);
}

constexpr inline size_t _StringLength(const char * const str) noexcept {
  return _StringLengthLoop(str, 0);
}

constexpr uint32_t _Murmur3Rotate_32(const uint32_t target,
    const uint8_t rotation) noexcept {
  return (target << rotation) | (target >> (32 - rotation));
}

constexpr uint32_t _Murmur3ShiftXor_32(const uint32_t hash, const size_t shift)
    noexcept {
  return hash ^ (hash >> shift);
}

constexpr uint32_t _Murmur3Last_32(const size_t len, const uint32_t hash)
    noexcept {
  return _Murmur3ShiftXor_32(0xc2b2ae35 * _Murmur3ShiftXor_32(
                        0x85ebca6b * _Murmur3ShiftXor_32(
                        hash ^ (uint32_t)len, 16), 13), 16);
}

constexpr uint32_t _Murmur3Tail1_32(const char data, const size_t len,
    const uint32_t hash, const uint32_t constant) noexcept {
  return _Murmur3Last_32(len, hash ^ (0x1b873593 * _Murmur3Rotate_32(
    0xcc9e2d51 * (constant ^ static_cast<uint8_t>(data)), 15)));
}

constexpr uint32_t _Murmur3Tail2_32(const char data[2], const size_t len,
    const uint32_t hash, const uint32_t constant) noexcept {
  return _Murmur3Tail1_32(data[0], len, hash,
    constant ^ (static_cast<uint8_t>(data[1]) << 8));
}

constexpr uint32_t _Murmur3Tail3_32(const char data[3], const size_t len,
    const uint32_t hash) noexcept {
  return _Murmur3Tail2_32(data, len, hash,
    0 ^ (static_cast<uint8_t>(data[2]) << 16));
}

constexpr uint32_t _Murmur3Rest_32(const char *const data,
    const size_t len, const uint32_t hash) noexcept {
  return ((len & 3) == 3) ? _Murmur3Tail3_32(&data[len - 3], len, hash) :
         ((len & 3) == 2) ? _Murmur3Tail2_32(&data[len - 2], len, hash, 0) :
         ((len & 3) == 1) ? _Murmur3Tail1_32(data[len - 1], len, hash, 0) :
                            _Murmur3Last_32(len, hash);
}

constexpr inline uint32_t _Murmur3Load_32(const char *const data,
    const size_t i) noexcept {
  return static_cast<uint32_t>(data[(i * sizeof(uint32_t)) + 3]) << 24 |
         static_cast<uint32_t>(data[(i * sizeof(uint32_t)) + 2]) << 16 |
         static_cast<uint32_t>(data[(i * sizeof(uint32_t)) + 1]) <<  8 |
         static_cast<uint32_t>(data[(i * sizeof(uint32_t)) + 0]) <<  0;
}

constexpr inline uint32_t _Murmur3Update_32(const uint32_t hash,
    const uint32_t update) noexcept {
  return 0xe6546b64 + (5 * _Murmur3Rotate_32(hash ^ (
    0x1b873593 * _Murmur3Rotate_32(0xcc9e2d51 * update, 15)), 13));
}

constexpr uint32_t _Murmur3Loop_32(const char *const data,
    const size_t len, const uint32_t hash, const size_t i = 0) noexcept {
  return (i  < (len / 4)) ?
          _Murmur3Loop_32(data, len,
            _Murmur3Update_32(hash, _Murmur3Load_32(data, i)), i + 1) :
          _Murmur3Rest_32(data, len, hash);
}

constexpr inline uint32_t Murmur3_32(const char *const key, const size_t length,
    const uint32_t seed) noexcept {
  return _Murmur3Loop_32(key, length, seed);
}

constexpr inline uint32_t Murmur3_32(const char *const str,
    const uint32_t seed = 0xAED123FD) noexcept {
  return Murmur3_32(str, _StringLength(str), seed);
}


