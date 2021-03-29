/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min::dataspace {

    class none : public dataspace_base {
    public:
        // the neutral unit is always a pass-through... compiler inlining should make it a noop
        class nothing {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                return input;
            }

            static inline number from_neutral(const number input) {
                return input;
            }
        };
    };

}    // namespace c74::min::dataspace
