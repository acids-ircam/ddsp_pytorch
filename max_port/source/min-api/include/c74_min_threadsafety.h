/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {

    /// There are several places where Min may check the thread of execution
    /// either for error catching or for altering the execution.
    /// This is true, for example, in the case of outlets.
    ///
    /// @seealso #thread_action
    /// @seealso #outlet<>

    enum class thread_check {
        main,         ///< Thread must be the main thread.
        scheduler,    ///< Thread must be the scheduler thread.
        any,          ///< Thread may be either of the main or the scheduler threads.
        none          ///< Perform no checking.
    };


    /// The thread action determines how a failure of a thread check is handled.
    ///
    /// @seealso #thread_check
    /// @seealso #outlet<>
    /// @seealso #fifo

    enum class thread_action {
        assert,    ///< Terminate execution
        fifo,      ///< Queue the operation(s) into a first-in-first-out buffer
        first,     ///< Queue the operation -- only queueing the first one if there are multiple
        last       ///< Queue the operation -- only queueing the last one if there are multiple
    };


    // Forward declaration... See below.

    template<class T, thread_check>
    class thread_trigger;


    // Forward declaration... See below.

    template<class T, thread_check check>
    void thread_trigger_callback(thread_trigger<T, check>* self);


    // The thread_trigger is a base class that triggers an action to be performed
    // in a subclass in the specified thread.
    // The subclass handles all details of the data storage / queueing / etc.
    // as well as interfacing with the code that needs the delivery of the triggered data.
    //
    // The thread_trigger is used in the implementation of outlets


    // The default thread_trigger is for the main thread / queue

    template<class T, thread_check check>
    class thread_trigger {
    public:
        // the baton is anything that will be handed off to be later handed back during the callback

        explicit thread_trigger(T a_baton)
        : m_baton{a_baton} {
            m_qelem = max::qelem_new(this, reinterpret_cast<max::method>(thread_trigger_callback<T, check>));
        }


        virtual ~thread_trigger() {
            max::qelem_free(m_qelem);
        }


        // Triggers cannot be copied.
        // If they are then the ownership of the internal t_clock/t_qelem becomes ambiguous.

        thread_trigger(const thread_trigger&) = delete;
        thread_trigger& operator=(const thread_trigger& value) = delete;


        // The baton is passed back later as an argument to the callback
        // @param a_baton The thing to store and then pass back when the trigger fires.

        void update_baton(T a_baton) {
            m_baton = a_baton;
        }


        // Tell the trigger to fire

        void set() {
            max::qelem_set(m_qelem);
        }


        // The thread_trigger interface requires both a
        // callback() for when the trigger fires
        // and a push() for handling new items.

        virtual void callback() = 0;
        virtual void push(const message_type a_type, const atoms& values) = 0;

    protected:
        T             m_baton;
        max::t_qelem* m_qelem;
    };


    // A thread_trigger specialization for the scheduler thread

    template<class T>
    class thread_trigger<T, thread_check::scheduler> {
    public:
        explicit thread_trigger(T a_baton)
        : m_baton{a_baton} {
            m_clock = max::clock_new(this, reinterpret_cast<max::method>(thread_trigger_callback<T, thread_check::scheduler>));
        }


        virtual ~thread_trigger() {
            object_free(m_clock);
        }


        // Triggers cannot be copied.
        // If they are then the ownership of the internal t_clock/t_qelem becomes ambiguous.

        thread_trigger(const thread_trigger&) = delete;
        thread_trigger& operator=(const thread_trigger& value) = delete;


        // The baton is passed back later as an argument to the callback
        // @param a_baton The thing to store and then pass back when the trigger fires.

        void update_baton(T a_baton) {
            m_baton = a_baton;
        }


        // Tell the trigger to fire

        void set() {
            max::clock_fdelay(m_clock, 0);
        }


        // The thread_trigger interface requires both a
        // callback() for when the trigger fires
        // and a push() for handling new items.

        virtual void callback() = 0;
        virtual void push(const message_type a_type, const atoms& values) = 0;

    protected:
        T             m_baton;
        max::t_clock* m_clock;
    };


    // C-style callback from the t_qelem or t_clock internal to a thread_trigger.
    // Simply forward the call back into the thread_trigger.

    template<class T, thread_check check>
    void thread_trigger_callback(thread_trigger<T, check>* self) {
        self->callback();
    }

}    // namespace c74::min
