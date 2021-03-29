# Copyright 2018 The Min-API Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

include(${C74_MAX_API_DIR}/script/max-posttarget.cmake)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

option(C74_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

if (APPLE)
    set(C74_XCODE_WARNING_CFLAGS "-Wall -Wmissing-field-initializers -Wno-unused-lambda-capture -Wno-unknown-warning-option")
    if (${C74_WARNINGS_AS_ERRORS})
        set(C74_XCODE_WARNING_CFLAGS "${C74_XCODE_WARNING_CFLAGS} -Werror")
    endif ()

    # enforce a strict warning policy
    set_target_properties(${PROJECT_NAME} PROPERTIES XCODE_ATTRIBUTE_WARNING_CFLAGS ${C74_XCODE_WARNING_CFLAGS})
endif ()
