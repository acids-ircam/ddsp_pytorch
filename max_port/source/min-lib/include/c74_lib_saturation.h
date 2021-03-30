/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_min_api.h"

namespace c74::min::lib {


    ///	A single-channel soft-saturation/distortion effect.

    class saturation {
    public:
        /// Set the amount of overdrive.
        /// @param	drive_percentage	The new amount of overdrive as a percentage.

        void drive(number drive_percentage) {
            m_drive = drive_percentage;
            auto f  = MIN_CLAMP(drive_percentage / 100.0, 0.001, 0.999);

            m_z    = M_PI * f;
            m_s    = 1.0 / sin(m_z);
            m_b    = MIN_CLAMP(1.0 / f, 0.0, 1.0);
            m_nb   = m_b * -1.0;
            auto i = int(f);
            if ((f - (double)i) > 0.5)
                m_scale = sin(m_z);    // sin(f * kTTPi);
            else
                m_scale = 1.0;
        }


        /// Return the current amout of overdriving being applied.
        /// @return The current overdrive amount as a percentage.

        number drive() {
            return m_drive;
        }


        /// Calculate one sample.
        ///	@return		Calculated sample

        sample operator()(sample x) {
            if (x > m_b)
                x = 1.0;
            else if (x < m_nb)
                x = -1.0;
#ifdef WIN_VERSION
            else {
                number sign;
                if (x < 0.0) {
                    x    = -x;
                    sign = -1.0;
                }
                else
                    sign = 1.0;
                x = sign * sin(m_z * x) * m_s;
            }
#else
            else
                x = sin(m_z * x) * m_s;
#endif
            return x * m_scale;
        }

    private:
        number m_drive{};
        number m_z;
        number m_s;
        number m_b;
        number m_nb;    // negative b
        number m_scale;
    };


}    // namespace c74::min::lib
