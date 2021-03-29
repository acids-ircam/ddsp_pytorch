/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {

    /// @defgroup buffers Buffer Objects

    /// A reference to a buffer~ object.
    /// The buffer_reference automatically adds the management hooks required for your object to work with a buffer~.
    /// This includes adding a 'set' message and a 'dblclick' message as well as dealing with notifications and binding.
    /// @ingroup buffers

    class buffer_reference {
    public:
        template<bool>
        friend class buffer_lock;

        static const constexpr int k_max_channels = 4096;    ///< The maximum number of channels supported by the buffer~ object.


        /// Create a reference to a buffer~ object.
        /// @param	an_owner	The owning object for the buffer reference. Typically you will pass `this`.
        /// @param	a_function	An optional function to be executed when the buffer reference issues notifications.
        ///						Typically the function is defined using a C++ lambda with the #MIN_FUNCTION signature.

        // takes a single arg, but cannot be marked explicit unless we are willing to decorate all using code with a cast to this type
        // thus we ignore the advice of C.46 @ https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md

        buffer_reference(object_base* an_owner, const function& a_function = nullptr, const bool create_messages = true)
        : m_owner { *an_owner }
        , m_notification_callback { a_function }
        {
            if (create_messages) {
                // Messages added to the owning object for this buffer~ reference

                m_set_meth = std::make_unique<message<>>(&m_owner, "set", "Choose a named buffer~ from which to read.",
                     MIN_FUNCTION {
                        set(args[0]);
                        return {};
                     }
                );

                m_dblclick_meth = std::make_unique<message<>>(&m_owner, "dblclick",
                     MIN_FUNCTION {
                         max::buffer_view(max::buffer_ref_getobject(m_instance));
                         return {};
                     }
                 );

                 m_notify_meth = std::make_unique<message<>>( &m_owner, "notify",
                     MIN_FUNCTION {
                         return handle_notification(&m_owner, args);
                    }
                 );
            }
        }

        // copy/move constructors and assignment
        buffer_reference(const buffer_reference& source) = delete;
        buffer_reference(buffer_reference&& source)      = delete;
        buffer_reference& operator=(const buffer_reference& source) = delete;
        buffer_reference& operator=(buffer_reference&& source) = delete;



        /// Destroy a buffer reference.

        ~buffer_reference() {
            object_free(m_instance);
        }


        /// Bind the buffer reference to a buffer with a different name.
        /// @param	name	The name of the buffer~ with which to bind the reference.

        void set(const symbol name) {
            if (!m_instance)
                m_instance = max::buffer_ref_new(m_owner, name);
            else
                buffer_ref_set(m_instance, name);
        }


        /// Call the buffer's notify method manually
        /// You will need to do this if you define a custom "notify" message for your object.

        atoms notify(const atoms& args) {
            return m_notification_callback(args, 0);
        }


        /// Find out if the buffer referenced actually exists
        /// @return	True if the named buffer~ exists. Otherwise false.

        operator bool() const {
            return m_instance && max::buffer_ref_exists(m_instance);
        }


        atoms handle_notification(object_base* an_owner, const atoms& args) {
            notification n { args };

            if (m_notification_callback) {
                if (n.name() == k_sym_globalsymbol_binding)
                    m_notification_callback({k_sym_binding}, -1);
                else if (n.name() == k_sym_globalsymbol_unbinding)
                    m_notification_callback({k_sym_unbinding}, -1);
                else if (n.name() == k_sym_buffer_modified)
                    m_notification_callback({k_sym_modified}, -1);
            }
            return { max::buffer_ref_notify(m_instance, n.registration(), n.name(), n.source(), n.data()) };
        }


    private:
        max::t_buffer_ref* m_instance { nullptr };
        object_base&       m_owner;
        function           m_notification_callback;

        // Messages added to the owning object for this buffer~ reference

        unique_ptr<message<>> m_set_meth {};
        unique_ptr<message<>> m_dblclick_meth {};
        unique_ptr<message<>> m_notify_meth {};
    };


    /// A lock guard and accessor for buffer~ access.
    ///	@tparam	audio_thread_access	Make this true if you will access the buffer~ from the audio thread.
    ///								Otherwise make this false for access on other threads.
    ///								The default is to access on the audio thread.
    /// @ingroup buffers

    template<bool audio_thread_access = true>
    class buffer_lock {
    public:
        /// Obtain buffer access from a buffer_reference
        /// @param	a_buffer_ref	The buffer reference to lock and thus gain access.

        buffer_lock(buffer_reference& a_buffer_ref);


        /// Return the lock to free up the buffer~ for access by others.

        ~buffer_lock();


        /// Determine if the buffer~ being accessed has valid samples to access.
        ///	@return	True if the buffer~ is valid and possesses samples. Otherwise false.

        bool valid() const {
            if (!m_buffer_obj || !m_tab)
                return false;
            else
                return true;
        }


        /// Determine the length of the buffer~ in samples.
        ///	@return	The length of the buffer~ in samples.
        /// @see	length_in_seconds()

        size_t frame_count() const {
            return max::buffer_getframecount(m_buffer_obj);
        }


        /// Determine the number of channels in the buffer~.
        ///	@return	The number of channels in the buffer~.

        size_t channel_count() const {
            return max::buffer_getchannelcount(m_buffer_obj);
        }


        /// Read or write the value of a specified sample in the buffer.
        /// @param index	The index to fetch the sample is into the memory of the buffer for all channels and frames.
        ///					The index is interleaved such that all samples for the first frame preceed all samples for the second frame, etc.
        ///	@return			A reference to the sample data for reading and/or writing.
        /// @see			lookup()

        float& operator[](long index) {
            return m_tab[index];
        }


        /// Read or write the value of a specified sample in the buffer.
        /// @param frame	The frame from which to fetch the sample reference.
        /// @param channel	The channel from which to fetch the sample reference.
        ///	@return			A reference to the sample data for reading and/or writing.

        float& lookup(size_t frame, size_t channel = 0) {
            if (frame >= frame_count())
                frame = frame_count() - 1;

            auto index = frame;

            if (channel_count() > 1)
                index = index * channel_count() + channel;

            return m_tab[index];
        }


        /// Determine the sample rate of the buffer~ contents.
        /// @return	The buffer~ sample rate.

        double samplerate() const {
            max::t_buffer_info info;

            max::buffer_getinfo(m_buffer_obj, &info);
            return info.b_sr;
        }


        /// Determine the length of the buffer~ in seconds.
        /// @return	The length of the buffer~ in seconds.
        /// @see	frame_count()

        double length_in_seconds() const {
            return frame_count() / samplerate();
        }


        /// Mark the buffer~ as dirty.
        /// This will notify other objects with a buffer reference that modifications have been made.
        /// For example, the waveform~ object relies on this to know that it must re-draw.

        void dirty() {
            max::buffer_setdirty(m_buffer_obj);
        }


        /// resize a buffer.
        /// only available for non-audio thread access.
        /// @param	length_in_ms	The new length to which the buffer should resize.

        template<bool U = audio_thread_access, typename enable_if<U == false, int>::type = 0>
        void resize(double length_in_ms) {
            max::object_attr_setfloat(m_buffer_obj, k_sym_size, length_in_ms);
        }


        /// resize a buffer.
        /// only available for non-audio thread access.
        /// @param	length_in_samples	The new length to which the buffer should resize.

        template<bool U = audio_thread_access, typename enable_if<U == false, int>::type = 0>
        void resize_in_samples(int length_in_samples) {
            max::t_atom_long newsize = length_in_samples;
            max::object_method(static_cast<max::t_object*>(m_buffer_obj), max::gensym("sizeinsamps"), (void*)newsize, 0);
        }

    private:
        buffer_reference&  m_buffer_ref;
        max::t_buffer_obj* m_buffer_obj { nullptr };
        float*             m_tab        { nullptr };
    };

}    // namespace c74::min
