/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#define C74_MIN_WITH_IMPLEMENTATION

#include "c74_min_api.h"
#include "c74_min_attribute_impl.h"
#include "c74_min_buffer_impl.h"
#include "c74_min_impl.h"

// The use of __has_include on Windows requires VS version 15.3 which is not yet available
// Alternatively defined C74_USE_MIN_LIB using CMake
#if defined(C74_USE_MIN_LIB)
    #include "../../min-lib/include/c74_lib.h"
#elif __has_include("../../min-lib/include/c74_lib.h")
    #include "../../min-lib/include/c74_lib.h"
#endif

#define UNUSED(x) ((void)x)

#undef C74_MIN_WITH_IMPLEMENTATION
