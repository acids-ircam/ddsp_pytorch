/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#ifdef WIN_VERSION
#pragma warning(push)
#pragma warning(disable : 26495)    // uninitialized member warning: because we use placement-new this warning is not appropriate
#endif


namespace c74::min {


    // forward declarations

    class inlet_base;
    class outlet_base;
    class argument_base;
    class message_base;
    class attribute_base;

    template<typename T, threadsafe threadsafety = threadsafe::undefined, template<typename> class limit_type = limit::none, allow_repetitions repetitions = allow_repetitions::yes>
    class attribute;


    // Wrap the C++ class together with the appropriate Max object header
    // Max object header is selected automatically using the type of the base class.

    template<class min_class_type, class = void>
    struct minwrap;


    // The header of instance's C-style struct. This is always a max::t_object.
    // It is sized such that it can accomodate the other extensions of a max::t_object as well.

    class maxobject_header {
    public:
        // allow casting the header to any of the max t_object struct types without errors/warnings

        operator max::t_object*() {
            return &m_objstorage.maxobj;
        }

        operator max::t_jbox*() {
            return &m_objstorage.maxbox;
        }

        operator max::t_pxobject*() {
            return &m_objstorage.mspobj;
        }

        operator max::t_pxjbox*() {
            return &m_objstorage.mspbox;
        }

        operator void*() {
            return &m_objstorage.maxobj;
        }

    private:
        union storage {
            max::t_object   maxobj;
            max::t_jbox     maxbox;
            max::t_pxobject mspobj;
            max::t_pxjbox   mspbox;
        };

        storage m_objstorage;
    };


    /// An object_base is a generic way to pass around a min::object.
    /// Required because a min::object<>, though sharing common code,
    /// is actually specific to the the user's defined class due to template specialization.

    class object_base {
        static const constexpr long k_magic = 1974004791;    // magic number used internally for sanity checking on pointers to objects

    protected:
        // Constructor is only called when creating a min::object<>, and never created directly.
        // The dictionary representing the object in the patcher is referenced (owned) by m_state.
        // Inheriting classes can retrieve information from this dictionary using the state() method.

        object_base()
        : m_state { (max::t_dictionary*)k_sym__pound_d, false }
        {}


        // Destructor is only called when freeing a min::object<>, and never directly.

        virtual ~object_base() {
            // TODO: free proxy inlets!
        }


    public:
        /// Is this class a Jitter class (e.g. a matrix_operator or gl_operator)?
        /// @return	True if it is. Otherwise false.

        virtual bool is_jitter_class() const = 0;


        /// Is this class a User Interface operator?
        /// @return True if it is. Otherwise false.

        virtual bool is_ui_class() const = 0;


        /// Does this class implement a "mousedragdelta" message?
        /// @return True if does. Otherwise false.

        virtual bool has_mousedragdelta() const = 0;


        /// Can an instance of this class capture the keyboard focus?
        /// @return True if it can. Otherwise false.

        virtual bool is_focusable() const = 0;


        /// Is this class assumed to have threadsafe attribute accessors and messages?
        /// @return True if it is. Otherwise false (the default).

        virtual bool is_assumed_threadsafe() const = 0;


        
        virtual strings tags() const = 0;


        /// Cast this object to it's corresponding t_object pointer as understood by the older C Max API.
        /// @return The t_object pointer for this object.

        operator max::t_object*() {
            return maxobj();
        }


        /// Cast this object to it's corresponding t_object pointer as understood by the older C Max API.
		/// @return The t_object pointer for this object.

		operator const max::t_object *() const {
			return maxobj();
		}


        /// Explicitly fetch this object's corresponding t_object pointer as understood by the older C Max API.
        /// @return The t_object pointer for this object.

        max::t_object* maxobj() {
            if (m_min_magic == k_magic)
                return m_maxobj;
            else
                return nullptr;
        }


        /// Explicitly fetch this object's corresponding t_object pointer as understood by the older C Max API.
		/// @return The t_object pointer for this object.

		const max::t_object* maxobj() const {
			if (m_min_magic == k_magic)
				return m_maxobj;
			else
				return nullptr;
		}


        /// Get a reference to this object's inlets.
        /// @return	A reference to this object's inlets.

        auto inlets() -> std::vector<inlet_base*>& {
            return m_inlets;
        }


        /// Get a reference to this object's outlets.
        /// @return	A reference to this object's outlets.

        auto outlets() -> std::vector<outlet_base*>& {
            return m_outlets;
        }


        /// Get a reference to this object's argument declarations.
        /// Note that to get the actual argument values you will need to call state() and parse the dictionary.
        /// @return	A reference to this object's argument declarations.

        auto arguments() const -> const std::vector<argument_base*>& {
            return m_arguments;
        }


        /// Get a reference to this object's messages.
        /// @return	A reference to this object's messages.

        auto messages() -> std::unordered_map<std::string, message_base*>& {
            return m_messages;
        }

        /// Get a reference to this object's messages.
        /// @return	A reference to this object's messages.

        auto messages() const -> const std::unordered_map<std::string, message_base*>& {
            return m_messages;
        }


        /// Get a reference to this object's attributes.
        /// @return	A reference to this object's attributes.

        auto attributes() -> std::unordered_map<std::string, attribute_base*>& {
            return m_attributes;
        }


        /// Get a reference to this object's attributes.
        /// @return	A reference to this object's attributes.

        auto attributes() const -> const std::unordered_map<std::string, attribute_base*>& {
            return m_attributes;
        }


        /// Is this object done being initialized?
        ///	@return	True if it is done with initialization and construction. Otherwise false.

        bool initialized() const {
            return m_initialized;
        }


        /// Get the dictionary representing this object's state in the patcher.
        /// @return	A dictionary with the object's state.
        /// @see	"Saving State" in GuideToWritingObjects.md

        dict state() {
            return m_state;
        }


        /// Return the name of the object as it was typed into the box to create this instance.
        /// Useful in the case where there may be one class with several aliases that modify the behavior (e.g. metro and qmetro).
        /// @return	The name of this class.

        symbol classname() const {
            return m_classname;
        }


        patcher patcher() {
            max::t_object* p {};

            auto err = max::object_obex_lookup(maxobj(), k_sym__pound_p, &p);
            if (err != max::MAX_ERR_NONE)
                error("unable to obtain owning patcher for object");
            return min::patcher(p);
        }


        box box() {
            max::t_object* b {};

            auto err = max::object_obex_lookup(maxobj(), k_sym__pound_b, &b);
            if (err != max::MAX_ERR_NONE)
                error("unable to obtain owning patcher for object");
            return min::box(b);
        }


        void attach(max::t_object* o, const symbol a_namespace = k_sym_nobox) {
            assert(o != nullptr);
            auto err = object_attach_byptr_register(maxobj(), o, a_namespace);
            if (err)
                error("cannot attach to object");
        }


        /// Try to call a named message.
        /// @param	name	The name of the message to attempt to call.
        /// @param	args	Any args you wish to pass to the message call.
        /// @return			If the message doesn't exist an empty set of atoms.
        ///					Otherwise the results of the message.

        atoms try_call(const std::string& name, const atoms& args = {});


        /// Try to call a named message.
        /// @param	name	The name of the message to attempt to call.
        /// @param	arg		A single atom arg you wish to pass to the message call.
        /// @return			If the message doesn't exist an empty set of atoms.
        ///					Otherwise the results of the message.

        atoms try_call(const std::string& name, const atom& arg) {
            atoms as = {arg};
            return try_call(name, as);
        }


        /// Find out if the object has a message with a specified name.
        /// @param	name		The name of the message to lookup.
        ///	@return				True if the object has a message with that name. Otherwise false.
        /// @see				try_call()

        bool has_call(const std::string& name) const {
            auto found_message = m_messages.find(name);
            return (found_message != m_messages.end());
        }

    private:
        max::t_object*                                   m_maxobj;       // initialized prior to placement new
        long                                             m_min_magic;    // should be valid if m_maxobj has been assigned
        bool                                             m_initialized { false };
        std::vector<inlet_base*>                         m_inlets;
        std::vector<outlet_base*>                        m_outlets;
        std::vector<argument_base*>                      m_arguments;
        std::unordered_map<std::string, message_base*>   m_messages;      // written at class init -- readonly thereafter
        std::unordered_map<std::string, attribute_base*> m_attributes;    // written at class init -- readonly thereafter
        dict                                             m_state;
        symbol                                           m_classname;    // what's typed in the max box

        friend class inlet_base;
        friend class outlet_base;

        friend class argument_base;

        template<class min_class_type, class>
        friend struct minwrap;

        template<class min_class_type>
        friend minwrap<min_class_type>* wrapper_new(const max::t_symbol* name, const long ac, const max::t_atom* av);

        template<class min_class_type>
        friend max::t_object* jit_new(const max::t_symbol* s);


        // Internal method called by the wrapper when creating an instance.
        // There are potentially two ways for objects to be instantiated:
        //
        // 1. instantiated by Max (using placement new in the wrapper code)
        // 2. instantiated some other way (not using placement new)
        //
        // When created by Max we need to have the member set for the
        // max object (m_maxobj) prior to the call to the constructor.
        // But, if we are instatiated directly then that memory is uninitialized.
        //
        // One option would be to use a global and access that during the construction.
        // That solution is wrought with many obvious problems.
        // This solution was chosen despite some different problems
        // (e.g. the rare case where the magic number would be randomly initialized to the correct value.)

        void assign_instance(max::t_object* instance) {
            m_maxobj    = instance;
            m_min_magic = k_magic;
        }


        // Called by the wrapper to create the Max counterparts to the Min inlets
        // after the Min object construction is complete.

        void create_inlets();


        // Called by the wrapper to create the Max counterparts to the Min outlets
        // after the Min object construction is complete.

        void create_outlets();


        // Called by the wrapper to indicate that it is done creating our instance.

        void postinitialize() {
            m_initialized = true;
        }


        // Called by the wrapper once the object is initialized so that it is possible to know what object name
        // was used to create this instance.
        // Useful in the case where there may be one class with several aliases that modify the behavior (e.g. metro and qmetro).

        void set_classname(const symbol s) {
            m_classname = s;
        }


        // Called by the min::argument to add an argument to the object.

        void register_argument(argument_base* arg) {
            m_arguments.push_back(arg);
        }

    public:
        // DO NOT USE
        // Intended to be private but made public to avoid excessive contortions required to make min_ctor<> a friend function
        // due to heavy use of templates, SFINAE, etc.
        //
        // Called by the wrapper to process all arguments after the object has been created
        // (but prior to attributes being processed)
        // defined in c74_min_argument.h

        void process_arguments(const atoms& args);
    };


    // The 'minwrap' is the struct for our Max object instance as we would think of it using the traditional Max SDK.
    // The first member is one of the variants of a t_object (via the maxobject_header).
    // Following that is a data member for an instance of our C++ Min class.
    //
    // All specializations of the minwrap struct must define how they are setup (when the instance is created)
    // and how they are torn down (when the instance is freed).
    // Much of this type of work is located elsewhere if it is possible to do so.
    // For example, argument processing is the same for all cases and thus it is present outside of this struct because it needs to
    // specialization.
    //
    // This (non-)specialization is a normal max object (which includes ui objects and jitter matrix operators).

    template<class min_class_type>
    struct minwrap<min_class_type, typename enable_if<!is_base_of<vector_operator_base, min_class_type>::value
                                                        && !is_base_of<mc_operator_base, min_class_type>::value
                                                        && !is_base_of<sample_operator_base, min_class_type>::value>::type> {
        maxobject_header m_max_header;
        min_class_type   m_min_object;


        // Setup is called at instantiation.

        void setup() {
            m_min_object.create_inlets();
            m_min_object.create_outlets();
        }


        // Cleanup is called when the instance is freed.

        void cleanup() {}


        // Enable passing the minwrap instance to Max C API calls without explicit casting or compiler warnings.

        max::t_object* maxobj() {
            return m_max_header;
        }
    };


    /// Deduce the intended name of a Max object from the name of the c++ sourcecode file.
    /// This is used internally via the #MIN_EXTERNAL macro when the max::t_class is created.
    /// Thus the maxname parameter may come in as an entire path because of use of the __FILE__ macro that is invoked.
    /// @param	maxname		The filename, possible the fullpath, of the c++ source file.
    /// @return				A string with the deduced object name as it will be exposed to the Max environment.

    inline std::string deduce_maxclassname(const char* maxname) {
        std::string smaxname;

        const char* start = strrchr(maxname, '/');    // mac paths
        if (start)
            start += 1;
        else {
            start = strrchr(maxname, '\\');    // windows paths
            if (start)
                start += 1;
            else
                start = maxname;
        }

        const char* end = strstr(start, "_tilde.cpp");    // audio objects
        if (end) {
            smaxname.assign(start, end - start);
            smaxname += '~';
        }
        else {    // all other objects
            const char* end = strrchr(start, '.');
            if (!end)
                end = start + strlen(start);
            if (!strcmp(end, ".cpp"))
                smaxname.assign(start, end - start);
            else
                smaxname = start;
        }
        return smaxname;
    }

}    // namespace c74::min

#ifdef WIN_VERSION
#pragma warning(pop)    // disable : 26495
#endif
