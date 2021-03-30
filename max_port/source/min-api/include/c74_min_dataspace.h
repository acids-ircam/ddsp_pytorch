/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min::dataspace {

    const double k_gain_midi_power { log(pow(10.0, 10.0 / 20.0)) / log(127.0 / 100.0) };
    const double k_gain_midi_power_r { 1.0 / k_gain_midi_power };

    class dataspace_base {
    public:
        // TODO: error checking -- can we do a static_assert that both source and dest are defined in the same dataspace?
        template<class source_unit_type, class dest_unit_type>
        static inline number convert(const number x) {
            return dest_unit_type::from_neutral(source_unit_type::to_neutral(x));
        }
    };

}

#include "c74_min_dataspace_gain.h"
#include "c74_min_dataspace_none.h"
#include "c74_min_dataspace_time.h"
