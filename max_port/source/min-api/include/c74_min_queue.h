/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    template<placeholder inlet_placeholder_type = placeholder::none>
    class queue;

    extern "C" void queue_qfn_callback(queue<>* a_queue);    // defined in c74_min_impl.h


    /// The queue class allows you to defer the call of a function to the near future in Max's main (low-priority) thread.
    ///
    /// @seealso	#timer
    ///	@seealso	#time_value
    /// @seealso	#fifo

    template<placeholder queue_placeholder_type>
    class queue {
    public:
        /// Create a queue.
        /// @param	an_owner	The owning object for the queue. Typically you will pass `this`.
        /// @param	a_function	A function to be executed when the queue is serviced.
        ///						Typically the function is defined using a C++ lambda with the #MIN_FUNCTION signature.

        queue(object_base* an_owner, const function a_function)
        : m_owner { an_owner }
        , m_function { a_function } {
            m_instance = max::qelem_new(this, reinterpret_cast<max::method>(queue_qfn_callback));
        }


        /// Destroy a timer.

        ~queue() {
            max::qelem_free(m_instance);
        }


        // Queues cannot be copied.
        // If they are then the ownership of the internal t_qelem becomes ambiguous.

        queue(const queue&) = delete;
        queue& operator=(const queue& value) = delete;


        /// Set the queue to fire the next time Max services main thread queues.
        /// When the queue fires its function will be executed.

        void set() {
            max::qelem_set(m_instance);
        }


        /// Calling a queue is the same as calling the set() method

        void operator()() {
            set();
        }


        /// Stop a queue that has been previously set().

        void unset() {
            max::qelem_unset(m_instance);
        }


        /// post information about the timer to the console
        // also serves the purpose of eliminating warnings about m_owner being unused

        void post() const {
            std::cout << m_instance << &m_function << m_owner << std::endl;
        }

    private:
        object_base*   m_owner;
        const function m_function;
        max::t_qelem*  m_instance { nullptr };

        friend void queue_qfn_callback(queue* a_queue);
        void        qfn() {
            atoms a;
            m_function(a, -1);
        }
    };


}    // namespace c74::min
