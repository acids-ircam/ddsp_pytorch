/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    /// A port represents a input or an output from an object.
    /// It is the base class for both #inlet and #outlet and is not intended to be created directly.
    ///
    /// @seealso #inlet
    /// @seealso #outlet

    class port {
    protected:
        port(object_base* an_owner, const std::string& a_description, const std::string& a_type)
        : m_owner(an_owner)
        , m_description(a_description)
        , m_type(a_type)
        {}

    public:
        /// Determine if an audio signal is currently connected to this port.
        ///	@return		True if a signal is connected. Otherwise false.

        bool has_signal_connection() const {
            return m_signal_connection;
        }


        /// Determine the type of the port.
        /// Most inlets and outlets are generic and thus the type is an empty string.
        /// Notable exceptions are "signal" for audio and "dictionary" for dictionaries.
        /// @return		A string with the type of the port, if a type has been given to the port.

        string type() const {
            return m_type;
        }


        /// Get the description of the port for either documentation generation
        /// or displaying assistance info in the UI.
        /// @return		A string with the description of the port.

        string description() const {
            return m_description;
        }

    protected:
        object_base* m_owner;
        string       m_description;
        string       m_type;
        bool         m_signal_connection { false };


        // update_signal_connection() is called to update our audio signal connection state
        // by Max's "dspsetup" method via min_dsp64_io()

        template<class min_class_type>
        friend void min_dsp64_io(minwrap<min_class_type>* self, const short* count);

        void update_signal_connection(const bool new_signal_connection_status) {
            m_signal_connection = new_signal_connection_status;
        }
    };


}    // namespace c74::min
