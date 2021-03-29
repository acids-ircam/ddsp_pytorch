/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2020 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    /// A notification
    /// Received from other objects when you are "attached" to them (listening)

    class notification {
    public:

        /// Represent message arguments as an instance of a notification
        ///
        /// @param  args    Five atoms passed as a notification in Max's object listener system, described as follows:
        /// * arg0    self
        /// * arg1    s         the registered name of the sending object
        /// * arg2    msg        then name of the notification/message sent
        /// * arg3    sender    the pointer to the sending object
        /// * arg4    data    optional argument sent with the notification/message

        notification(const atoms& args) {
            if (args.size() != 5)
                error("incorrect number of arguments for notification");
            m_self = args[0];
            m_sender_name = args[1];
            m_notification_name = args[2];
            m_sender = args[3];
            m_data = args[4];
        }


        /// Is this notification for a local attribute being modified?
        /// @return true if is a local attribute modified notification, otherwise false.

        bool is_attr_modified() const {
            return (m_sender == m_self) && (m_notification_name == k_sym_attr_modified);
        }


        /// The name of the attribute being modified if this is a notification for local attribute modification.
        /// @return the name of the attribute if this notification is for a local attribute notification. otherwise the empty symbol will be returned.

        auto attr_name() const {
            symbol attribute_name;
            if (is_attr_modified()) {
                auto retval = c74::max::object_method(m_data, k_sym_getname);
                attribute_name = static_cast<c74::max::t_symbol*>(retval);
            }
            return attribute_name;
        }


        /// The name of the notification message
        /// @return the name of the notification

        auto name() const {
            return m_notification_name;
        }


        /// The sender of the notification message
        /// @return A pointer to the sender of the notification.

        auto source() const {
            return m_sender;
        }


        /// The payload or data for the notification, if any.
        /// @return a pointer to the data of the notification.

        auto data() const {
            return m_data;
        }


        /// The sending object's registered name in Max's object attachment system.
        /// @return the internally-registered name of the sending object.

        auto registration() const {
            return m_sender_name;
        }
        

    private:
        max::t_object*  m_self;
        symbol          m_sender_name;
        symbol          m_notification_name;
        max::t_object*  m_sender;
        max::t_object*  m_data;
    };


}    // namespace c74::min
