/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "catch.hpp"    // The Catch header must come first -- otherwise some C++ includes will tangle with it and cause problems.
#include "c74_min.h"    // The standard Min header


// epsilon value used for comparing floats for approximate equality

constexpr auto g_min_epsilon = std::numeric_limits<float>::epsilon()*1000;


/// Wrapper for Catch's Appox() method with more accomating defaults, particularly for comparing values near zero.

#define APPROX(x) Approx(x).epsilon(g_min_epsilon).margin(g_min_epsilon)


/// Compare a container (e.g. a vector) of floats in a Catch unit test.
/// If there is a failure, not only will the return value be false, but a Catch REQUIRE will fail.
///
/// @tparam T           The type for comparisons. Should be an STL container of floats or doubles.
/// @param  source      The values to compare
/// @param  reference   The reference values against which to compare.
/// @return             True if they are (approximately) the same. Otherwise false.

template<typename T>
bool require_vector_approx(const T source, const T reference) {
    REQUIRE(source.size() == reference.size());
    if (source.size() != reference.size())
        return false;

    for (auto i = 0; i < source.size(); ++i) {
        INFO("when i == " << i);
        REQUIRE(source[i] == APPROX(reference[i]));
        if (source[i] != APPROX(reference[i]))
            return false;
    }
    return true;
}


/// Compare a container (e.g. a vector) of floats in a Catch unit test.
/// If there is a failure, not only will the return value be false, but a Catch REQUIRE will fail.
///
/// @tparam T           The type for comparisons. Should be an STL container of floats or doubles.
/// @param  source      The values to compare
/// @param  reference   The reference values against which to compare.
/// @return             True if they are (approximately) the same. Otherwise false.

#define REQUIRE_VECTOR_APPROX(source, reference) require_vector_approx(source, reference)
