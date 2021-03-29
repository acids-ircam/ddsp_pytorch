/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2020 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    /// A mouse or touch event

    class event {
    public:
        enum class input_type {
            mouse = 1,
            touch,
            pen
        };


        event(const atoms& args)
        : m_target { args }
        {
            if (args.size() == 1) {
                // a single atom indicates we are being passed a pointer to an event
                auto* e = static_cast<event*>( static_cast<void*>(args[0]) );
                *this = *e;
            }
            else if (args.size() != 5) {
                error("incorrect number of arguments for notification");
            }
            else {
                m_self = args[0];
                //m_target = args[1];
                m_x = args[2];
                m_y = args[3];
                m_modifiers = args[4];
            }
        }


        event(max::t_object* o, max::t_object* a_patcherview, const max::t_mouseevent& a_max_mouseevent)
        : m_target { o, a_patcherview }
        {
            m_self            = o;
			m_index           = static_cast<int>( a_max_mouseevent.index );
            m_x               = a_max_mouseevent.position.x;
            m_y               = a_max_mouseevent.position.y;
            m_modifiers       = a_max_mouseevent.modifiers;
            m_type            = static_cast<input_type>( a_max_mouseevent.type );
			m_pen_pressure    = a_max_mouseevent.pressure;
			m_pen_orientation = a_max_mouseevent.orientation;
			m_pen_rotation    = a_max_mouseevent.rotation;
			m_pen_tilt_x      = a_max_mouseevent.tiltX;
			m_pen_tilt_y      = a_max_mouseevent.tiltY;
        }


        /// Is this notification for a local attribute being modified?
        /// @return true if is a local attribute modified notification, otherwise false.

        bool is_command_key_down() const {
            return (m_modifiers & c74::max::eCommandKey);
        }

        bool is_shift_key_down() const {
            return (m_modifiers & c74::max::eShiftKey);
        }


        /// The name of the notification message
        /// @return the name of the notification

        auto target() const {
            return m_target;
        }


        auto index() const {
			return m_index;
        }


        /// The sender of the notification message
        /// @return A pointer to the sender of the notification.

        auto x() const {
            return m_x;
        }


        /// The payload or data for the notification, if any.
        /// @return a pointer to the data of the notification.

        auto y() const {
            return m_y;
        }


        auto type() const {
            return m_type;
        }

        auto pen_pressure() const {
			return m_pen_pressure;
        }

		auto pen_orientation() const {
			return m_pen_orientation;
        }
		
        auto pen_rotation() const {
			return m_pen_rotation;
        }
		
        auto pen_tilt_x() const {
			return m_pen_tilt_x;
        }
		
        auto pen_tilt_y() const {
			return m_pen_tilt_y;   
        }

    private:
        max::t_object*  m_self;
        ui::target      m_target;
		int             m_index {};
		number          m_x {};
		number          m_y {};
		int             m_modifiers {};
		input_type      m_type {};
		number          m_pen_pressure {};
		number          m_pen_orientation {};
		number          m_pen_rotation {};
		number          m_pen_tilt_x {};
		number          m_pen_tilt_y {};
    };


}    // namespace c74::min
