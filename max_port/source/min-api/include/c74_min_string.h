/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min::str {


    /// Trim leading and trailing whitespace from a string
    /// @param	s	The string to trim
    /// @return		The trimmed string

    inline string trim(const string& s) {
        if (!s.empty()) {
            size_t first = s.find_first_not_of(' ');
            size_t last  = s.find_last_not_of(' ');
            return s.substr(first, (last - first + 1));
        }
        else
            return s;
    }


    /// Split a string into a vector of substrings on a specified delimiter
    /// @param	s		The string to split
    /// @param	delim	The delimiter on which to split the string
    /// @return			A vector of substrings

    inline vector<string> split(const string& s, const char delim) {
        vector<string>    substrings;
        string            substring;
        std::stringstream ss(s);

        while (getline(ss, substring, delim))
            substrings.push_back(substring);
        return substrings;
    }


    /// Concatenate N strings using a definable "glue" character
    /// @param	input	A vector of all strings to be joined together into a single string.
    /// @param	glue	The character to use inbetween each string in the input.
    /// @return			The single concatenated string.

    inline string join(const vector<string>& input, const char glue = ' ') {
        string output;

        for (const auto& str : input) {
            output += str;
            output += glue;
        }
        trim(output);
        return output;
    }

}    // namespace c74::min::str
