/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once


namespace c74::min::lib::filters {


    /// Utility: generate an impulse response from a set of coefficients
    /// @param	a	Feedforward coefficients (numerator).
    /// @param	b	Feedback coefficients (denominator).
    /// @param	N	Optional size of the generated response, default 64.
    ///	@return		A vector of samples the generated impulse response of the specified filter.

    inline auto generate_impulse_response(const sample_vector& a, const sample_vector& b, int N = 64) {
        sample_vector x(N);    // input -- feedforward history
        sample_vector y(N);    // output -- feedback history

        std::fill_n(x.begin(), N, 0.0);
        std::fill_n(y.begin(), N, 0.0);
        x[0] = 1.0;

        for (int n = 0; n < N; n++) {
            for (auto i = 0; i < a.size(); i++) {
				auto j = n - i;
                if (j < 0)
                    y[n] += 0.0;
                else
                    y[n] += (a[i] * x[j]);
            }

            // ignore b[0] and assume it is normalized to 1.0
            for (auto i = 1; i < b.size(); i++) {
				auto j = n - i;
                if (j < 0)
                    y[n] -= 0.0;
                else
                    y[n] -= (b[i] * y[j]);
            }
        }
        return y;
    }


}    // namespace c74::min::lib::filters
