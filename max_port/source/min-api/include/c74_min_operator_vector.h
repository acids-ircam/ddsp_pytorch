/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    /// An audio bundle is a container for N channels of M-sized vectors of audio sample values.

    struct audio_bundle {

        /// Create an audio bundle from an existing array-array of sample values.
        /// This is used internally by Min when receiving the per-vector callback from Max for audio objects
        /// to create an audio_bundle to pass to a vector_operator class.
        /// @param	samples			A pointer to the first sample in the first channel in an array of channels of vectors of audio.
        /// @param	channel_count	The number of channels in the memory pointed to by the samples parameter.
        /// @param	frame_count		The size (in samples) of the audio vectors for each channel.

        audio_bundle(double** samples, const long channel_count, const long frame_count)
        : m_samples { samples }
        , m_channel_count { channel_count }
        , m_frame_count { frame_count }
        {}


        /// Get a direct pointer to the memory of the audio bundle.
        /// @return	A pointer to the beginning memory location of the audio bundle.

        double** samples() {
            return m_samples;
        }


        /// Get a pointer to the samples for a specific channel.
        /// @param	channel		The channel for which to fetch the pointer.
        ///						NOTE: No bounds checking is performed!
        ///						You are responsible for only accessing valid channels!
        /// @return				A pointer the beginning memory location for samples in the specified channel.

        double* samples(const size_t channel) {
            return m_samples[channel];
        }


        /// Determine the number of channels in an audio bundle.
        /// @return		The number of channels in the audio bundle.

        // The return type is a long because that is what the callback from Max provides us.
        // While it not ideal we also do not want to spend computational cycles casting it to a size_t either.

        long channel_count() const {
            return m_channel_count;
        }


        /// Determine the number of samples in each vector of an audio bundle.
        /// @return		The number of frames in the audio bundle.

        long frame_count() const {
            return m_frame_count;
        }


        /// Zero-out the data in the entire audio bundle.

        void clear() {
            for (auto channel = 0; channel < m_channel_count; ++channel) {
                for (auto i = 0; i < m_frame_count; ++i)
                    m_samples[channel][i] = 0.0;
            }
        }


        /// Copy an audio_bundle to another audio_bundle without resizing the destination
        ///
        /// If the destination does not have enough channels to copy the entire source
        /// the extra channels will be dropped.
        ///
        /// If the destination has too many channels (and we would leave them in a non-determinant state)
        /// then assert (crash).
        ///
        /// Similarly, both source and destination are REQUIRED to have the same framesize or an assertion will fire.
        ///
        /// @param 	other	The audio_bundle that will be the source of the copy.
        /// @return			The destination to which the contents of the audio bundle is copied.

        audio_bundle& operator=(const audio_bundle& other) {
            assert(m_channel_count <= other.m_channel_count);
            assert(m_frame_count == other.m_frame_count);

            for (auto channel = 0; channel < m_channel_count; ++channel) {
                for (auto i = 0; i < m_frame_count; ++i)
                    m_samples[channel][i] = other.m_samples[channel][i];
            }
            return *this;
        }

    private:
        double** m_samples { nullptr };
        long     m_channel_count {};
        long     m_frame_count {};
    };


    // A specialization of "minwrap" (the container of the Max t_object together with the Min class)
    // for audio objects (both vector_operator and sample_operator)
    //
    // The generic version of minwrap is defined in c74_min_object_components.h

    template<class min_class_type>
    struct minwrap<min_class_type, type_enable_if_audio_class<min_class_type>> {
        maxobject_header m_max_header;
        min_class_type   m_min_object;


        // Setup is called at instantiation.

        void setup() {
            max::dsp_setup(m_max_header, (long)m_min_object.inlets().size());

            if (m_min_object.is_ui_class()) {
                max::t_pxjbox* x = m_max_header;
                x->z_misc |= max::Z_NO_INPLACE;
                if (is_base_of<mc_operator_base, min_class_type>::value)
                    x->z_misc |= max::Z_MC_INLETS;
            }
            else {
                max::t_pxobject* x = m_max_header;
                x->z_misc |= max::Z_NO_INPLACE;
                if (is_base_of<mc_operator_base, min_class_type>::value)
                    x->z_misc |= max::Z_MC_INLETS;
            }

            m_min_object.create_outlets();
        }


        // Cleanup is called when the object is freed.

        void cleanup() {
            if (m_min_object.is_ui_class())
                max::dsp_freejbox(m_max_header);
            else
                max::dsp_free(m_max_header);
        }


        // Enable passing the minwrap instance to Max C API calls without explicit casting or compiler warnings.

        max::t_object* maxobj() {
            return m_max_header;
        }
    };


    // Represents any specialized type of vector_operator<>.
    //
    // All of the available operator extensions to min::object<> classes are implemented using a model of a "base" class
    // that can be used generically and the actual operator type which is specialized with templates.
    // Some of the operator types don't actually require the templates at this time but all operators are implemented this way
    // both for consistency and to allow template parameters to be added in the future without breaking existing code.

    class vector_operator_base {};


    /// Inherit from vector_operator to extend your class for processing vectors of audio samples.
    /// This offers the most flexible efficient way to write audio objects.
    /// The caveat is that there is greater potential for subtle errors and the efficiency of
    /// coding and maintenance is less ideal than if you inherit from sample_operator<>.
    ///
    /// In some cases inheriting from vector_operator<> will be more computationally efficient.
    /// This is particularly true if your object will perform buffer~ access.
    ///
    /// @tparam vector_operator_placeholder_type	Unused. You should supply no arguments. For example, `vector_operator<>`.
    /// @see sample_operator
    /// @see buffer_reference

    template<placeholder vector_operator_placeholder_type = placeholder::none>
    class vector_operator : public vector_operator_base {
    public:
        ///	Set a new samplerate.
        /// You will not typically have any need to call this.
        /// It is called internally any time the dsp chain containing your object is compiled.
        /// @param	a_samplerate	A new samplerate with which your object will be updated.

        void samplerate(const double a_samplerate) {
            m_samplerate = a_samplerate;
        }


        /// Return the current samplerate for this object's signal chain.
        /// @return	The samplerate in hz.

        double samplerate() const {
            return m_samplerate;
        }


        ///	Set a new vector size.
        /// You will not typically have any need to call this.
        /// It is called internally any time the dsp chain containing your object is compiled.
        /// @param	a_vector_size	A new vector size with which your object will be updated.

        void vector_size(const double a_vector_size) {
            m_vector_size = a_vector_size;
        }


        /// Return the current vector size for this object's signal chain.
        /// @return	The vector size in samples.

        double vector_size() const {
            return m_vector_size;
        }


        /// Process one sample of audio using the derived class.
        /// This will call your Min object's operator for processing vectors with a vector size of 1.
        /// It is primarily useful for unit testing.
        /// @param	x	The input sample.
        /// @return		The output sample.

        sample operator()(sample x);


        /// All classes extending vector_operator<> must implement this operator
        /// which is responsible for processing audio when callbacks come from Max's signal chain.
        /// @param	input	The incoming audio.
        /// @param	output	The outgoing audio.

        virtual void operator()(audio_bundle input, audio_bundle output) = 0;

    private:
        double  m_samplerate { c74::max::sys_getsr() };        // initialized to the global samplerate, but updated to the local samplerate when the dsp chain is compiled.
        int     m_vector_size { c74::max::sys_getblksize() };  // ...
    };


    // The performer class wraps the C callback routine for a Max audio "perform" method.
    // It adapts the calls coming from the Max application to the call operator implemented in the Min class.
    // The correct version of this enabled using SFINAE template enabling depending on whether this is a
    // vector_operator<> or one of the sample_operator<> extending class.
    //
    // The sample_operator<> versions are implemented in c74_min_operator_sample.h

    template<class min_class_type, class enable = void>
    class performer {
    public:
        // The traditional Max audio "perform" callback routine

        static void perform(minwrap<min_class_type>* self, max::t_object* dsp64, double** in_chans, const long numins, double** out_chans, const long numouts, const long sampleframes, const long, const void*) {
            audio_bundle input {in_chans, numins, sampleframes};
            audio_bundle output {out_chans, numouts, sampleframes};
            self->m_min_object(input, output);
        }
    };


    // SFINAE implementation used internally to determine if the Min class has a member named dspsetup.
    // NOTE: This relies on the C++ member name being "dspsetup" -- not just the Max message name being "dspsetup".
    // See the min.buffer.loop~ object for an example.
    //
    // To test this in isolation, for a class named slide, use the following code:
    // static_assert(has_dspsetup<slide>::value, "error");

    template<typename min_class_type>
    struct has_dspsetup {
        template<class, class>
        class checker;

        template<typename C>
        static std::true_type test(checker<C, decltype(&C::dspsetup)>*);

        template<typename C>
        static std::false_type test(...);

        typedef decltype(test<min_class_type>(nullptr)) type;
        static const bool value = is_same<std::true_type, decltype(test<min_class_type>(nullptr))>::value;
    };

    // An alternative to the above to all "m_dspsetup" in addition to "dspsetup"

    template<typename min_class_type>
    struct has_m_dspsetup {
        template<class, class>
        class checker;

        template<typename C>
        static std::true_type test(checker<C, decltype(&C::m_dspsetup)>*);

        template<typename C>
        static std::false_type test(...);

        typedef decltype(test<min_class_type>(nullptr)) type;
        static const bool value = is_same<std::true_type, decltype(test<min_class_type>(nullptr))>::value;
    };


    // The "dsp64" method for Max audio objects is split up into several components here.
    // The main "dsp64" method is min_dsp64(), which needs to obey basic C rules because it is called by Max.
    // This in-turn then calls min_dsp64_sel() which is a templated C++ function that is specialized based on the properties of the Min
    // class. Each of those specializations needs to perform some common/shared functions which are then factored out as well.

    // The min_dsp64_io function handles updating the inlet and outlet connection state any time the dsp64 message is called.

    template<class min_class_type>
    void min_dsp64_io(minwrap<min_class_type>* self, const short* count) {
        int i = 0;

        while (i < self->m_min_object.inlets().size()) {
            self->m_min_object.inlets()[i]->update_signal_connection(count[i] != 0);
            ++i;
        }
        while (i < self->m_min_object.outlets().size()) {
            self->m_min_object.outlets()[i - self->m_min_object.inlets().size()]->update_signal_connection(count[i] != 0);
            ++i;
        }
    }


    template<class min_class_type, enable_if_vector_operator<min_class_type> = 0>
    void min_dsp64_attrmap(minwrap<min_class_type>* self, const short* count)
    {}


    // The min_dsp64_add_perform function handles adding the perform method to the signal chain (see performer class above)

    template<class min_class_type>
    void min_dsp64_add_perform(minwrap<min_class_type>* self, max::t_object* dsp64) {
        // find the perform method and add it
        object_method_direct(void, (max::t_object*, max::t_object*, const max::t_perfroutine64, const long, const void*), dsp64, symbol("dsp_add64"),
            self->maxobj(), reinterpret_cast<max::t_perfroutine64>(performer<min_class_type>::perform), 0, NULL);
    }


    // A specialization of min_dsp64_sel for classes that have a custom "dspsetup" message.


    template<class min_class_type>
    typename enable_if<has_dspsetup<min_class_type>::value
    || has_m_dspsetup<min_class_type>::value>::type
    min_dsp64_sel(minwrap<min_class_type>* self, max::t_object* dsp64, const short* count, const double samplerate, const long maxvectorsize, const long flags) {
        self->m_min_object.samplerate(samplerate);
        self->m_min_object.vector_size(maxvectorsize);
        min_dsp64_io(self, count);
        min_dsp64_attrmap(self, count);

        atoms args;
        args.push_back(atom(samplerate));
        args.push_back(atom(max::t_atom_long(maxvectorsize)));
        self->m_min_object.dspsetup(args);

        min_dsp64_add_perform(self, dsp64);
    }


    // A (non)specialization of min_dsp64_sel for classes that do _not_ have a custom "dspsetup" message
    // (which is most audio classes).

    template<class min_class_type>
    typename enable_if<!has_dspsetup<min_class_type>::value
    && !has_m_dspsetup<min_class_type>::value>::type
    min_dsp64_sel(minwrap<min_class_type>* self, max::t_object* dsp64, const short* count, const double samplerate, const long maxvectorsize, const long flags) {
        self->m_min_object.samplerate(samplerate);
        self->m_min_object.vector_size(maxvectorsize);
        min_dsp64_io(self, count);
        min_dsp64_attrmap(self, count);
        min_dsp64_add_perform(self, dsp64);
    }


    // The dsp64 method that interfaces with the call from Max when compiling the signal chain.

    template<class min_class_type>
    type_enable_if_audio_class<min_class_type>
    min_dsp64(minwrap<min_class_type>* self, max::t_object* dsp64, const short* count, const double samplerate, const long maxvectorsize, const long flags) {
        min_dsp64_sel<min_class_type>(self, dsp64, count, samplerate, maxvectorsize, flags);
    }


    // Add audio support to a Max external when the max::t_class is being setup.
    // A call to wrap_as_max_external_audio() will be called for all externals when wrapping the Min class.
    // Only in cases where the class is actually and audio class (inherits from vector_operator or sample_operator)
    // will this this specialization be called.

    template<class min_class_type, enable_if_audio_class<min_class_type> = 0>
    void wrap_as_max_external_audio(max::t_class* c) {
        max::class_addmethod(c, reinterpret_cast<max::method>(min_dsp64<min_class_type>), "dsp64", max::A_CANT, 0);
        if (is_base_of<ui_operator_base, min_class_type>::value)
            max::class_dspinitjbox(c);
        else
            max::class_dspinit(c);
    }

}    // namespace c74::min
