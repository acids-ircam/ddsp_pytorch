/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    /// The base class for all template specializations of sample_operator.

    class mc_operator_base {};


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
    /// @tparam vector_operator_placeholder_type Unused.

    template<placeholder vector_operator_placeholder_type = placeholder::none>
    class mc_operator : public mc_operator_base {
    public:

        /// Default constructor.

        mc_operator() {}


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
        double m_samplerate{c74::max::sys_getsr()};    // initialized to the global samplerate, but updated to the local samplerate when the
                                                       // dsp chain is compiled.
        int m_vector_size{c74::max::sys_getblksize()};    // ...
        vector<std::pair<int,attribute_base*>> m_attributes_mapped_to_inlets;
    };

    template<class min_class_type, enable_if_mc_operator<min_class_type> = 0>
    void min_dsp64_attrmap(minwrap<min_class_type>* self, const short* count) {}

}    // namespace c74::min
