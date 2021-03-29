/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    // Type definition for what the legacy C Max SDK uses to represent an outlet.

    using t_max_outlet = void*;


    // outlet_do_send() wraps all Max API function calls for all types.
    //
    // Code that uses outlets send their data out of the outlet using a send() method.
    // There are various factors that determine how this call gets translated into a call to Max's legacy API:
    //
    // * The type of data being sent (int, float, symbol, list, etc.)
    // * The thread specification which indicates which thread(s) are valid targets for the outlet call.
    // * The thread action which indicates if/how to check the thread and what to do if it doesn't match the specification.
    //
    // Regardless of how the threading is handled, when it comes time to actually send the data to the outlet the
    // outlet_do_send() helper function is called.

    template<typename outlet_type>
    inline void outlet_do_send(const t_max_outlet maxoutlet, const outlet_type& value) {
        if (value[0].a_type == max::A_LONG || value[0].a_type == max::A_FLOAT)
            max::outlet_list(maxoutlet, nullptr, static_cast<short>(value.size()), static_cast<const max::t_atom*>(&value[0]));
        else {
            if (value.size() > 1)
                max::outlet_anything(maxoutlet, value[0], static_cast<short>(value.size() - 1), static_cast<const max::t_atom*>(&value[1]));
            else
                max::outlet_anything(maxoutlet, value[0], 0, nullptr);
        }
    }

    template<>
    inline void outlet_do_send<max::t_atom_long>(const t_max_outlet maxoutlet, const max::t_atom_long& value) {
        max::outlet_int(maxoutlet, value);
    }

    template<>
    inline void outlet_do_send<double>(const t_max_outlet maxoutlet, const double& value) {
        max::outlet_float(maxoutlet, value);
    }


#ifdef MAC_VERSION
#pragma mark -
#endif


    // Calls to send() on an outlet may deliver the data to the Max outlet in one of several ways.
    // If the call is made on the thread that is specified in the outlet's Thread Specification
    // then the call is executed immediately and synchronously.
    // If the call is made on a different thread than what is specified in the outlet's Thread Specification
    // then this outlet_queue class is invoked to handle the Thread Action.
    //
    // The possible actions to take when the thread is different than the specification are implemented below:
    //
    // * assert:	The default action, crash when this condition occurs.
    // * first:		Schedule or defer the call to the appropriate thread,
    //				if multiple sends are received before the call is serviced then only send the first one.
    // * last:		Schedule or defer the call to the appropriate thread,
    //				if multiple sends are received before the call is serviced then only send the last one.
    // * fifo:		Schedule or defer the call to the appropriate thread,
    //				if multiple sends are received before the call is serviced they are put in a fifo and all are delivered.
    //
    // The outlet_queue inherits from thread_trigger.
    // The thread_trigger manages the internal t_qelem or t_clock used to trigger the callback.


    // ASSERT: default thread_action is to assert, which means no queue at all...

    template<thread_check check, thread_action action>
    class outlet_queue : public thread_trigger<t_max_outlet, check> {
    public:
        explicit outlet_queue(const t_max_outlet a_maxoutlet)
        : thread_trigger<t_max_outlet, check>(a_maxoutlet)
        {}

        void callback() {}

        void push(const message_type, const atoms&) {}
    };


    // FIRST: store only the first value and discard additional values (opposite of usurp)

    template<thread_check check>
    class outlet_queue<check, thread_action::first> : public thread_trigger<t_max_outlet, check> {
    public:
        explicit outlet_queue(const t_max_outlet a_maxoutlet)
        : thread_trigger<t_max_outlet, check>(a_maxoutlet)
        {}

        void callback() {
            outlet_do_send(m_value);
            m_set = false;
        }

        void push(const message_type a_type, const atoms& as) {
            if (!m_set) {
                m_value = as;
                m_set   = true;
                thread_trigger<t_max_outlet, check>::set();
            }
        }

    private:
        atoms m_value;
        bool  m_set { false };
    };


    // LAST: store only the last value received (usurp)

    template<thread_check check>
    class outlet_queue<check, thread_action::last> : public thread_trigger<t_max_outlet, check> {
    public:
        explicit outlet_queue(const t_max_outlet a_maxoutlet)
        : thread_trigger<t_max_outlet, check>(a_maxoutlet)
        {}

        void callback() {
            outlet_do_send(this->m_maxoutlet, m_value);
        }

        void push(const message_type a_type, const atoms& as) {
            m_value = as;
            thread_trigger<t_max_outlet, check>::set();
        }

    private:
        atoms m_value;
    };


    // FIFO: defer all values

    template<thread_check check>
    class outlet_queue<check, thread_action::fifo> : public thread_trigger<t_max_outlet, check> {

        struct tagged_atoms {
            message_type m_type;
            atoms        m_as;
        };

    public:
        explicit outlet_queue(const t_max_outlet a_maxoutlet)
        : thread_trigger<t_max_outlet, check>(a_maxoutlet)
        {}

        void callback() {
            tagged_atoms tas;
            while (m_values.try_dequeue(tas)) {
                if (tas.m_type == message_type::int_argument)
                    outlet_do_send<max::t_atom_long>(this->m_baton, tas.m_as[0]);
                else if (tas.m_type == message_type::float_argument)
                    outlet_do_send<double>(this->m_baton, tas.m_as[0]);
                else
                    outlet_do_send(this->m_baton, tas.m_as);
            }
        }

        void push(const message_type a_type, const atoms& as) {
            tagged_atoms tas{a_type, as};
            m_values.enqueue(tas);
            thread_trigger<t_max_outlet, check>::set();
        }


    private:
        fifo<tagged_atoms> m_values;
    };


#ifdef MAC_VERSION
#pragma mark -
#endif


    // Checks the currently executing thread at runtime against the thread specified as a template argument.
    // Returns true if they are the same, otherwise returns false.

    template<thread_check>
    bool outlet_call_is_safe();


    // Forward declarations of the outlet.
    // Default parameters:
    //
    // * We do not check threads in release builds unless specifically requested (for performance reasons).
    // * We do check it in debug builds though unless it is specifically requested not to check.

#ifdef NDEBUG
    template<thread_check check = thread_check::none, thread_action action = thread_action::assert>
    class outlet;
#else    // DEBUG
    template<thread_check check = thread_check::any, thread_action action = thread_action::assert>
    class outlet;
#endif


    // FIRST / LAST / FIFO: queue up the data if the outlet call is not thread-safe

    template<thread_check check_type, thread_action action_type, typename outlet_type>
    class handle_unsafe_outlet_send {
    public:
        handle_unsafe_outlet_send(outlet<check_type, thread_action::fifo>* an_outlet, const outlet_type& a_value) {
            if (typeid(outlet_type) == typeid(max::t_atom_long))
                an_outlet->queue_storage().push(message_type::int_argument, a_value);
            else if (typeid(outlet_type) == typeid(double))
                an_outlet->queue_storage().push(message_type::float_argument, a_value);
            else    // atoms
                an_outlet->queue_storage().push(message_type::gimme, a_value);
        }
    };


    // ASSERT: throw an assertion if the outlet call is not thread-safe

    template<thread_check check_type, typename outlet_type>
    class handle_unsafe_outlet_send<check_type, thread_action::assert, outlet_type> {
    public:
        handle_unsafe_outlet_send(outlet<check_type, thread_action::assert>* an_outlet, const outlet_type& a_value) {
            assert(false);
        }
    };


#ifdef MAC_VERSION
#pragma mark -
#endif


    // Represents any type of outlet.
    // Used internally to allow heterogenous containers of outlets.

    class outlet_base : public port {
        friend void object_base::create_outlets();

    public:
        outlet_base(object_base* an_owner, const string& a_description, const string& a_type)
        : port(an_owner, a_description, a_type)
        {}

        virtual ~outlet_base() {}

    private:
        virtual void create() = 0;

    protected:
        t_max_outlet m_instance { nullptr };
    };


    /// An outlet for sending output from a Min object.
    /// Outlets are specialized with two optional template parameters.
    /// In most cases you will use the defaults. For example, `outlet<>`.
    ///
    /// @tparam check	Define which threads are valid sources for calls to the outlet.
    /// @tparam action	Define what action is to be taken if the thread check fails.

    template<thread_check check, thread_action action>
    class outlet : public outlet_base {

        // utility: queue an argument of any type for output

        template<typename argument_type>
        void queue_argument(const argument_type& arg) noexcept {
            m_accumulated_output.push_back(arg);
        }

        // utility: empty argument handling (required for all recursive variadic templates)

        void handle_arguments() noexcept {
            ;
        }

        // utility: handle N arguments of any type by recursively working through them
        //	and matching them to the type-matched routine above.

        template<typename FIRST_ARG, typename... REMAINING_ARGS>
        void handle_arguments(FIRST_ARG const& first, REMAINING_ARGS const&... args) noexcept {
            queue_argument(first);
            if (sizeof...(args))
                handle_arguments(args...);    // recurse
        }

    public:
        /// Create an outlet
        /// @param an_owner			The Min object instance that owns this outlet. Typically you should pass 'this'.
        /// @param a_description	Documentation string for this outlet.
        /// @param a_type			Optional string defining the Max message type of the outlet for checking patch-cord connections.
        /// @param an_atom_count	Optional number of atoms that will be passed out as a list from this outlet.
        ///							When greater than 1, defining this allows memory to be pre-allocated to improve performance.

        outlet(object_base* an_owner, const string& a_description, const string& a_type = "", const size_t an_atom_count = 1)
        : outlet_base(an_owner, a_description, a_type) {
            m_owner->outlets().push_back(this);
            m_accumulated_output.reserve(an_atom_count);
        }


        /// Create an outlet
        /// @param an_owner			The Min object instance that owns this outlet. Typically you should pass 'this'.
        /// @param a_description	Documentation string for this outlet.
        /// @param an_atom_count	Optional number of atoms that will be passed out as a list from this outlet.
        ///							When greater than 1, defining this allows memory to be pre-allocated to improve performance.
        /// @param a_type			Optional string defining the Max message type of the outlet for checking patch-cord connections.

        outlet(object_base* an_owner, const string& a_description, const size_t an_atom_count, const string& a_type = "")
        : outlet(an_owner, a_description, a_type, an_atom_count)
        {}


        /// Send a value out an outlet
        /// @param value The value to send.

        void send(const bool value) {
            if (outlet_call_is_safe<check>())
                outlet_do_send(m_instance, (max::t_atom_long)value);
            else
                handle_unsafe_outlet_send<check, action, max::t_atom_long>(this, value);
        }


        /// Send a value out an outlet
        /// @param value The value to send.

        void send(const int value) {
            if (outlet_call_is_safe<check>())
                outlet_do_send(m_instance, (max::t_atom_long)value);
            else
                handle_unsafe_outlet_send<check, action, max::t_atom_long>(this, value);
        }


        /// Send a value out an outlet
        /// @param value The value to send.

        void send(const long value) {
            if (outlet_call_is_safe<check>())
                outlet_do_send(m_instance, (max::t_atom_long)value);
            else
                handle_unsafe_outlet_send<check, action, max::t_atom_long>(this, value);
        }


        /// Send a value out an outlet
        /// @param value The value to send.

        void send(const size_t value) {
            if (outlet_call_is_safe<check>())
                outlet_do_send(m_instance, (max::t_atom_long)value);
            else
                handle_unsafe_outlet_send<check, action, max::t_atom_long>(this, value);
        }


        /// Send a value out an outlet
        /// @param value The value to send.

        void send(const float value) {
            if (outlet_call_is_safe<check>())
                outlet_do_send(m_instance, (double)value);
            else
                handle_unsafe_outlet_send<check, action, double>(this, value);
        }


        /// Send a value out an outlet
        /// @param value The value to send.

        void send(const double value) {
            if (outlet_call_is_safe<check>())
                outlet_do_send(m_instance, (double)value);
            else
                handle_unsafe_outlet_send<check, action, double>(this, value);
        }


        /// Send values out an outlet
        /// @param value The values to send.

        void send(const atoms& value) {
            if (value.empty())
                return;

            if (outlet_call_is_safe<check>())
                outlet_do_send(m_instance, value);
            else
                handle_unsafe_outlet_send<check, action, atoms>(this, value);
        }


        /// Send values out an outlet
        /// @param args The values to send.

        template<typename... ARGS>
        void send(ARGS... args) {
            handle_arguments(args...);
            send(m_accumulated_output);
            m_accumulated_output.clear();
        }


        /// Send values out an outlet
        /// @param args The values to send.

        template<typename... ARGS>
        void operator()(ARGS... args) {
            send(args...);
        }

    private:
        atoms                       m_accumulated_output;
        outlet_queue<check, action> m_queue_storage { this->m_instance };


        // called by object_base::create_outlets() when the owning object is constructed

        void create() override {
            if (type() == "")
                m_instance = max::outlet_new(m_owner->maxobj(), nullptr);
            else
                m_instance = max::outlet_new(m_owner->maxobj(), type().c_str());
            m_queue_storage.update_baton(m_instance);
        }


        // outlet_queue is drained by handle_unsafe_outlet_send()

        template<thread_check check_type, thread_action action_type, typename outlet_type>
        friend class handle_unsafe_outlet_send;

        outlet_queue<check, action>& queue_storage() {
            return m_queue_storage;
        }
    };


}    // namespace c74::min
