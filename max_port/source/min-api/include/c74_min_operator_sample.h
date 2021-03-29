/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    /// The base class for all template specializations of sample_operator.

    class sample_operator_base {};


    /// Inheriting from sample_operator extends your class functionality to processing audio
    /// by calculating samples one at a time using the call operator member of your class.
    ///
    /// Your call operator must take the same number of parameters as the input_count template arg.
    /// Additionally, your call operator must return an array of samples of the same size as the output_count template arg.
    /// For example, if your object inherits from sample_operator<3,2> then your call operator will be prototyped as:
    /// @code
    /// samples<2> operator() (sample input1, sample input2, sample input3);
    /// @endcode
    ///
    /// @tparam input_count_param		The number of audio inputs for your object.
    /// @tparam output_count_param	The number of audio outputs for your object.

    template<size_t input_count_param, size_t output_count_param>
    class sample_operator : public sample_operator_base {
    public:

        /// Default constructor.

        sample_operator() {}


        /// Get pointers to the attributes mapped to audio inlets

        auto& mapped_attributes() {
            return m_attributes_mapped_to_inlets;
        }


        /// Return the number of audio inputs for this sample operator class.
        /// @return The number of audio inputs.

        static constexpr size_t input_count() {
            return input_count_param;
        }


        /// Return the number of audio outputs for this sample operator class.
        /// @return The number of audio outputs.

        static constexpr size_t output_count() {
            return output_count_param;
        }


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


        // Ideally we would also declare a pure virtual function call operator
        // for the inheriting class to implement.
        // That is impossible, however, because we can't generically prototype N arguments
        // where N = output_count_param.
        //
        // Some example prototypes:
        //
        // sample operator() (sample input);
        // samples<2> operator() (sample input1, sample input2, sample input3);
        // void operator() (sample input1, sample input2);

    private:
        double m_samplerate {c74::max::sys_getsr()};    // initialized to the global samplerate, but updated to the local samplerate when the
                                                       // dsp chain is compiled.
        int m_vector_size {c74::max::sys_getblksize()};    // ...
        vector<std::pair<int,attribute_base*>> m_attributes_mapped_to_inlets;
    };


    template<class min_class_type, enable_if_sample_operator<min_class_type> = 0>
    void min_dsp64_attrmap(minwrap<min_class_type>* self, const short* count) {
        auto& attrs { self->m_min_object.mapped_attributes() };
        auto& inlets { self->m_min_object.inlets() };

        attrs.clear();

        for (auto i=0; i<inlets.size(); ++i) {
            auto& inlet = inlets[i];
            if (inlet->has_signal_connection() && inlet->has_attribute_mapping())
                attrs.push_back( { i, inlet->attribute() } );
        }
    }


    // To implement the performer class (below) generically we use std::array<sample> for both input and output.
    // However, we wish to define the call operator in the Min class with each sample as a
    // separate argument.
    // To make this translation compute efficiently and without out lots of duplicated code we use a pattern whereby
    // the sequence of indices for std::array are generated at compile time and then used to make the call
    // as a variadic template function.
    //
    // for more information, see:
    // http://stackoverflow.com/questions/16834851/passing-stdarray-as-arguments-of-template-variadic-function

    namespace detail {
        template<int... Is>
        struct seq {};

        template<int N, int... Is>
        struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

        template<int... Is>
        struct gen_seq<0, Is...> : seq<Is...> {};
    }    // namespace detail


    // A container of N samples, one for each inlet, that then makes a call to be processed by a sample_operator<>.

    template<class min_class_type, int count>
    struct callable_samples {

        explicit callable_samples(minwrap<min_class_type>* a_self)
        : self(a_self)
        {}

        void set(const size_t index, const sample& value) {
            data[index] = value;
        }

        auto call() {
            return call(detail::gen_seq<count>());
        }

        template<int... Is>
        auto call(detail::seq<Is...>) {
            return self->m_min_object(data[Is]...);
        }

        samples<count>           data;
        minwrap<min_class_type>* self;
    };


    // version of perform_copy_output() for samples<N> returned by the sample_operator<>'s call operator in the performer below.

    template<class min_class_type, typename type_returned_from_call_operator>
    void perform_copy_output(minwrap<min_class_type>* self, const size_t index, double** out_chans, const type_returned_from_call_operator vals) {
        for (auto chan = 0; chan < self->m_min_object.output_count(); ++chan)
            out_chans[chan][index] = vals[chan];
    }


    // version of perform_copy_output() for a single sample returned by the sample_operator<>'s call operator in the performer below.

    template<class min_class_type>
    void perform_copy_output(minwrap<min_class_type>* self, const size_t index, double** out_chans, const sample val) {
        out_chans[0][index] = val;
    }


    // The performer class wraps the C callback routine for a Max audio "perform" method.
    // It adapts the calls coming from the Max application to the call operator implemented in the Min class.
    // The correct version of this enabled using SFINAE template enabling depending on whether this is a
    // vector_operator<> or one of the sample_operator<> extending class.
    //
    // For sample_operator<> there are several versions of this wrapping/adapting callback.
    // This one is optimized for the most common case: a single input and a single output.

    template<class min_class_type>
    class performer<min_class_type, typename enable_if<is_base_of<sample_operator<1, 1>, min_class_type>::value>::type> {
    public:
        // The traditional Max audio "perform" callback routine

        static void perform(minwrap<min_class_type>* self, max::t_object* dsp64, const double** in_chans, const long numins, double** out_chans, const long numouts, const long sampleframes, const long, const void*) {
            auto in_samps  = in_chans[0];
            auto out_samps = out_chans[0];

            for (auto i = 0; i < sampleframes; ++i) {
                auto in      = in_samps[i];
                auto out     = self->m_min_object(in);
                out_samps[i] = out;
            }
        }
    };


    // The performer class wraps the C callback routine for a Max audio "perform" method.
    // This specialization is for a single input with no outputs

    template<class min_class_type>
    class performer<min_class_type, typename enable_if<is_base_of<sample_operator<1, 0>, min_class_type>::value>::type> {
    public:
        // The traditional Max audio "perform" callback routine

        static void perform(minwrap<min_class_type>* self, max::t_object* dsp64, const double** in_chans, const long numins, double** out_chans, const long numouts, const long sampleframes, const long, const void*) {
            auto in_samps = in_chans[0];

            for (auto i = 0; i < sampleframes; ++i) {
                auto in = in_samps[i];
                self->m_min_object(in);
            }
        }
    };


    // The performer class wraps the C callback routine for a Max audio "perform" method.
    // This is the generic version of the min_performer class, for N inputs and N outputs.
    // See above for other specializations.

    template<class min_class_type>
    class performer<min_class_type,
        typename enable_if<is_base_of<sample_operator_base, min_class_type>::value
                        && !is_base_of<sample_operator<1, 1>, min_class_type>::value
                        && !is_base_of<sample_operator<1, 0>, min_class_type>::value>::type> {
    public:
        static void perform(minwrap<min_class_type>* self, max::t_object* dsp64, const double** in_chans, const long numins, double** out_chans, const long numouts, const long sampleframes, const long, const void*) {
            auto& attrs { self->m_min_object.mapped_attributes() };
            const auto input_count { self->m_min_object.input_count() };

            if (attrs.empty()) {

                // the typical case:

                for (auto i = 0; i < sampleframes; ++i) {
                    callable_samples<min_class_type, min_class_type::input_count()> ins(self);

                    for (auto chan = 0; chan < input_count; ++chan)
                        ins.set(chan, in_chans[chan][i]);

                    auto out = ins.call();

                    if (numouts > 0)
                        perform_copy_output(self, i, out_chans, out);
                }
            }
            else {

                // the case where audio inlets are mapped to attributes

                for (auto i = 0; i < sampleframes; ++i) {
                    callable_samples<min_class_type, min_class_type::input_count()> ins(self);

                    for (auto& inletnum_and_attr : attrs) {
                        int 			inletnum { inletnum_and_attr.first };
                        attribute_base*	attr { inletnum_and_attr.second };
                        auto			value { in_chans[inletnum][i] };
                        atoms			a {{value}};

                        attr->set(a, false, false);
                    }

                    for (auto chan = 0; chan < input_count; ++chan)
                        ins.set(chan, in_chans[chan][i]);

                    auto out = ins.call();

                    if (numouts > 0)
                        perform_copy_output(self, i, out_chans, out);
                }
            }
        }
    };

}    // namespace c74::min
