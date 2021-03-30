/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    /// A standard callback function used throughout Min for various purposes.
    /// Typically this is provided to argument as a lamba function using the #MIN_FUNCTION macro.
    /// @param	as		A vector of atoms which may contain any arguments passed to your function.
    /// @param	inlet	The number (zero-based index) of the inlet at which the message was received, if relevant. Otherwise -1.
    /// @see		MIN_FUNCTION

    using function = std::function<atoms(const atoms& as, const int inlet)>;


    /// Provide the correct lamba function prototype for the min::argument constructor.
    /// @see argument
    /// @see argument_function

    #define MIN_FUNCTION [this](const c74::min::atoms& args, const int inlet) -> c74::min::atoms


    // Represents any type of message.
    // Used internally to allow heterogenous containers of messages for the Min class.

    class message_base {
    protected:
        // Constructor. See the constructor documention for min::message<> to get more details on the arguments.

        message_base(object_base* an_owner, const std::string& a_name, const function& a_function, const description& a_description = {}, const message_type type = message_type::gimme)
        : m_owner { an_owner }
        , m_function { a_function }
        , m_type { type }
        , m_description { a_description } {
            assert(m_function != nullptr);    // could happen if a function is passed as the arg but that fn hasn't initialized yet

            std::string name = a_name;

            if (name == "integer") {
                name   = "int";
                m_type = message_type::int_argument;
            }
            else if (name == "number") {
                name   = "float";
                m_type = message_type::float_argument;
            }
            else if (a_name == "dblclick" || a_name == "dsp64" || a_name == "dspsetup" || a_name == "edclose" || a_name == "fileusage"
                || a_name == "jitclass_setup" || a_name == "maxclass_setup" || a_name == "maxob_setup" || a_name == "mop_setup"
                || a_name == "notify" || a_name == "okclose" || a_name == "oksize" || a_name == "paint"
                || a_name == "patchlineupdate" || a_name == "savestate" || a_name == "setup"
                || a_name == "mouseenter" || a_name == "mouseleave" || a_name == "mousedown" || a_name == "mouseup" || a_name == "mousemove"
                || a_name == "mousedragdelta" || a_name == "mousedoubleclick"
                || a_name == "focusgained" || a_name == "focuslost"
                || a_name == "loadbang")
            {
                m_type = message_type::cant;
            }

            strings tags = an_owner->tags();

            auto tag_iter = std::find(tags.begin(), tags.end(), "multitouch");
            if (tag_iter != tags.end()) {
                if (a_name == "mouseenter")
                    name = "mt_mouseenter";
                else if (a_name == "mousemove")
                    name = "mt_mousemove";
                else if (a_name == "mousedown")
                    name = "mt_mousedown";
                else if (a_name == "mousedrag")
                    name = "mt_mousedrag";
                else if (a_name == "mouseup")
                    name = "mt_mouseup";
                else if (a_name == "mouseleave")
                    name = "mt_mouseleave";
            }

            m_name                    = name;
            m_owner->messages()[name] = this;    // add the message to the owning object's pile
        }

    public:
        // All messages must define what happens when you call them.

        virtual atoms operator()(const atoms& args = {}, const int inlet = -1) = 0;
        virtual atoms operator()(const atom arg, const int inlet = -1)        = 0;


        /// Return the Max C API message type constant for this message.
        /// @return The type of the message as a numeric constant.

        long type() const {
            return static_cast<long>(m_type);
        }


        /// Casting a message to a message_type allows easy checking of a message's type.
        /// @return The type of the message

        operator message_type() const {
            return m_type;
        }


        /// Return the provided description for use in documentation generation, auto-complete, etc.
        /// @return	The description string supplied when the message was created.

        std::string description_string() const {
            return m_description;
        }


        /// Return the name of the message.
        /// @return	The symbolic name of the message.

        symbol name() const {
            return m_name;
        }

    protected:
        object_base* m_owner;
        function     m_function;
        message_type m_type { message_type::gimme };
        symbol       m_name;
        description  m_description;

        friend class object_base;

        void update_inlet_number(int& inlet) {
            if (inlet == -1 && m_owner->maxobj()) {
                if (m_owner->inlets().size() > 1)    // avoid this potentially expensive call if there is only one inlet
                    inlet = static_cast<int>(proxy_getinlet(static_cast<max::t_object*>(*m_owner)));
                else
                    inlet = 0;
            }
        }
    };


    template<threadsafe threadsafety>
    class message;


    class deferred_message {
    public:
        deferred_message(message<threadsafe::no>* an_owning_message, const atoms& args, const int inlet)
        : m_owning_message{an_owning_message}
        , m_args{args}
        , m_inlet{inlet}
        {}


        deferred_message()
        {}


        // call a message's action and remove it from the queue
        // will free the deferred_message in the process

        void pop();


    private:
        message<threadsafe::no>* m_owning_message { nullptr };
        atoms                    m_args {};
        int                      m_inlet { -1 };
    };


    /// A message.
    /// Messages (sometimes called Methods) in Max are how actions are triggered in objects.
    /// When you create a message in your Min class you provide the action that it should trigger as an argument,
    /// usually using a lambda function and the #MIN_FUNCTION macro.
    ///
    /// By default, all messages are assumed to not be threadsafe and thus will defer themselves to the main thread if input has come from
    /// another thread. This behavior can be modified using the optional template parameter. DO NOT pass threadsafe::yes as the template
    /// parameter unless you are truly certain that you have made your code threadsafe.
    ///
    /// @tparam		threadsafety	If your object has been written specifically and carefully to be threadsafe then you may pass the option
    /// parameter
    ///								threadsafe::yes.
    ///								Otherwise you should just accept the default and let Min handle the threadsafety for you.

    template<threadsafe threadsafety = threadsafe::undefined>
    class message : public message_base {
    public:
        /// Create a new message for a Min class.
        ///
        /// @param	an_owner		The Min object instance that owns this outlet. Typically you should pass 'this'.
        /// @param	a_name			The name of the message. This is how users in Max will trigger the message action.
        /// @param	a_function		The function to be called when the message is received by your object.
        ///							This is typically provided as a lamba function using the #MIN_FUNCTION definition.
        /// @param	a_description	Optional, but highly encouraged, description string to document the message.
        /// @param	a_type			Optional message type determines what kind of messages Max can send.
        ///							In most cases you should _not_ pass anything here and accept the default.

        message(object_base* an_owner, const std::string& a_name, const function& a_function, const description& a_description = {}, const message_type a_type = message_type::gimme)
        : message_base(an_owner, a_name, a_function, a_description, a_type)
        {}


        /// Create a new message for a Min class.
        ///
        /// @param	an_owner		The Min object instance that owns this outlet. Typically you should pass 'this'.
        /// @param	a_name			The name of the message. This is how users in Max will trigger the message action.
        /// @param	a_description	Optional, but highly encouraged, description string to document the message.
        /// @param	a_function		The function to be called when the message is received by your object.
        ///							This is typically provided as a lamba function using the #MIN_FUNCTION definition.

        message(object_base* an_owner, const std::string& a_name, const description& a_description, const function& a_function)
        : message_base(an_owner, a_name, a_function, a_description)
        {}


        /// Create a new message for a Min class.
        ///
        /// @param	an_owner		The Min object instance that owns this outlet. Typically you should pass 'this'.
        /// @param	a_name			The name of the message. This is how users in Max will trigger the message action.
        /// @param	a_description	Optional, but highly encouraged, description string to document the message.
        /// @param	a_type			Optional message type determines what kind of messages Max can send.
        /// @param	a_function		The function to be called when the message is received by your object.
        ///							This is typically provided as a lamba function using the #MIN_FUNCTION definition.

        message(object_base* an_owner, const std::string& a_name, const description& a_description, message_type a_type, const function& a_function)
        : message(an_owner, a_name, a_function, a_description, a_type)
        {}

        virtual ~message() {}

        /// Call the message's action.
        /// @param	args	Optional arguments to send to the message's action.
        /// @return			Any return values will be returned as atoms.

        atoms operator()(const atoms& args = {}, const int an_inlet = -1) override {
			int inlet {an_inlet};
            update_inlet_number(inlet);

            // this is the same as what happens in a defer() call
            if (m_owner->is_assumed_threadsafe() || max::systhread_ismainthread())
                return m_function(args, inlet);
            else {
                deferred_message m { reinterpret_cast<message<threadsafe::no>*>(this), args, inlet };
                if (m_deferred_messages.try_enqueue(m)) {
                    max::defer(this->m_owner->maxobj(), reinterpret_cast<max::method>(message<threadsafety>::defer_callback),
                               reinterpret_cast<c74::max::t_symbol*>(this), 0, nullptr);
                }
            }
            return {};
        }


        void pop() {
            auto r = m_deferred_messages.peek();
            r->pop();
        }

        static void defer_callback(max::t_object* self, message<threadsafety>* m, const long, max::t_atom*) {
            m->pop();
        }


        /// Call the message's action.
        /// @param	arg		A single argument to send to the message's action.
        /// @return			Any return values will be returned as atoms.

        atoms operator()(const atom arg, const int inlet = -1) override {
            atoms as { arg };
            return (*this)(as, inlet);
        }

    private:
        // Any messages received from outside the main thread will be deferred using the queue below.

        friend class deferred_message;
        fifo<deferred_message> m_deferred_messages { 2 };
    };


    // specialization of message for messages which declare themselves to be not threadsafe

    template<>
    class message<threadsafe::no> : public message_base {
    public:
        /// Create a new message for a Min class.
        ///
        /// @param	an_owner		The Min object instance that owns this outlet. Typically you should pass 'this'.
        /// @param	a_name			The name of the message. This is how users in Max will trigger the message action.
        /// @param	a_function		The function to be called when the message is received by your object.
        ///							This is typically provided as a lamba function using the #MIN_FUNCTION definition.
        /// @param	a_description	Optional, but highly encouraged, description string to document the message.
        /// @param	type			Optional message type determines what kind of messages Max can send.
        ///							In most cases you should _not_ pass anything here and accept the default.

        message(object_base* an_owner, const std::string& a_name, const function& a_function, const description& a_description = {}, const message_type type = message_type::gimme)
        : message_base(an_owner, a_name, a_function, a_description)
        {}


        /// Create a new message for a Min class.
        ///
        /// @param	an_owner		The Min object instance that owns this outlet. Typically you should pass 'this'.
        /// @param	a_name			The name of the message. This is how users in Max will trigger the message action.
        /// @param	a_description	Optional, but highly encouraged, description string to document the message.
        /// @param	a_function		The function to be called when the message is received by your object.
        ///							This is typically provided as a lamba function using the #MIN_FUNCTION definition.

        message(object_base* an_owner, const std::string& a_name, const description& a_description, const function& a_function)
        : message_base(an_owner, a_name, a_function, a_description)
        {}


        /// Call the message's action.
        /// @param	args	Optional arguments to send to the message's action.
        /// @return			Any return values will be returned as atoms.

        atoms operator()(const atoms& args = {}, const int an_inlet = -1) override {
			int inlet {an_inlet};
            update_inlet_number(inlet);

            // this is the same as what happens in a defer() call
            if (max::systhread_ismainthread())
                return m_function(args, inlet);
            else {
                deferred_message m { this, args, inlet };
                if (m_deferred_messages.try_enqueue(m)) {
                    max::defer(this->m_owner->maxobj(), reinterpret_cast<max::method>(message<threadsafe::no>::defer_callback),
                               reinterpret_cast<c74::max::t_symbol*>(this), 0, nullptr);
                }
            }
            return {};
        }

        void pop() {
            auto r = m_deferred_messages.peek();
            r->pop();
        }

        static void defer_callback(max::t_object* self, message<threadsafe::no>* m, long, const max::t_atom*) {
            m->pop();
        }


        /// Call the message's action.
        /// @param	arg		A single argument to send to the message's action.
        /// @return			Any return values will be returned as atoms.

        atoms operator()(const atom arg, const int inlet = -1) override {
            atoms as { arg };
            return (*this)(as, inlet);
        }

    private:
        // Any messages received from outside the main thread will be deferred using the queue below.

        friend class deferred_message;
        fifo<deferred_message> m_deferred_messages { 2 };
    };


    // specialization of message for messages which declare themselves to be threadsafe

    template<>
    class message<threadsafe::yes> : public message_base {
    public:
        /// Create a new message for a Min class.
        ///
        /// @param	an_owner		The Min object instance that owns this outlet. Typically you should pass 'this'.
        /// @param	a_name			The name of the message. This is how users in Max will trigger the message action.
        /// @param	a_function		The function to be called when the message is received by your object.
        ///							This is typically provided as a lamba function using the #MIN_FUNCTION definition.
        /// @param	a_description	Optional, but highly encouraged, description string to document the message.
        /// @param	a_type			Optional message type determines what kind of messages Max can send.
        ///							In most cases you should _not_ pass anything here and accept the default.

        message(object_base* an_owner, const std::string& a_name, const function& a_function, const description& a_description = {}, const message_type a_type = message_type::gimme)
        : message_base(an_owner, a_name, a_function, a_description, a_type)
        {}


        /// Create a new message for a Min class.
        ///
        /// @param	an_owner		The Min object instance that owns this outlet. Typically you should pass 'this'.
        /// @param	a_name			The name of the message. This is how users in Max will trigger the message action.
        /// @param	a_description	Optional, but highly encouraged, description string to document the message.
        /// @param	a_function		The function to be called when the message is received by your object.
        ///							This is typically provided as a lamba function using the #MIN_FUNCTION definition.

        message(object_base* an_owner, const std::string& a_name, const description& a_description, const function& a_function)
        : message_base(an_owner, a_name, a_function, a_description)
        {}


        /// Call the message's action.
        /// @param	args	Optional arguments to send to the message's action.
        /// @param	inlet	Optional inlet number associated with the incoming message.
        /// @return			Any return values will be returned as atoms.

        atoms operator()(const atoms& args = {}, const int an_inlet = -1) override {
			int inlet {an_inlet};
            update_inlet_number(inlet);
            return m_function(args, inlet);
        }


        /// Call the message's action.
        /// @param	arg		A single argument to send to the message's action.
        /// @param	inlet	Optional inlet number associated with the incoming message.
        /// @return			Any return values will be returned as atoms.

        atoms operator()(const atom arg, const int inlet = -1) override {
            return m_function({arg}, inlet);
        }
    };

}    // namespace c74::min
