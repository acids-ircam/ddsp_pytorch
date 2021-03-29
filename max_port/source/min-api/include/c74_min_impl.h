/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_min_api.h"

namespace c74::min {


#ifdef __APPLE__
#pragma mark object_base
#endif


    // implemented out-of-line because of bi-directional dependency of min::message<> and min::object_base

    atoms object_base::try_call(const std::string& name, const atoms& args) {
        auto found_message = m_messages.find(name);
        if (found_message != m_messages.end())
            return (*found_message->second)(args);
        return {};
    }


    // implemented out-of-line because of bi-directional dependency of min::argument<> and min::object_base

    void object_base::process_arguments(const atoms& args) {
        auto arg_count = std::min(args.size(), m_arguments.size());

        for (auto i = 0; i < arg_count; ++i)
            (*m_arguments[i])(args[i]);
    }


    // inlets have to be created as a separate step (by the wrapper) because
    // max creates them from right-to-left
    // note that some objects will not call this function... i.e. dsp objects or other strongly-typed objects.

    void object_base::create_inlets() {
        if (m_inlets.empty())
            return;
        for (auto i = m_inlets.size() - 1; i > 0; --i)
            m_inlets[i]->m_instance = max::proxy_new(m_maxobj, static_cast<long>(i), nullptr);
    }


    // outlets have to be created as a separate step (by the wrapper) because
    // max creates them from right-to-left

    void object_base::create_outlets() {
        for (auto outlet = m_outlets.rbegin(); outlet != m_outlets.rend(); ++outlet)
            (*outlet)->create();
    }


#ifdef __APPLE__
#pragma mark -
#pragma mark c-style callbacks
#endif


    // c-style callback from the max kernel (clock for the min::timer class)

    void timer_tick_callback(timer<>* a_timer) {
        if (a_timer->should_defer())
            a_timer->defer();
        else
            a_timer->tick();
    }


    // c-style callback from the max kernel (qelem for the min::timer class)

    void timer_qfn_callback(timer<>* a_timer) {
        a_timer->tick();
    }


    // c-style callback from the max kernel (qelem for the min::queue class)

    void queue_qfn_callback(queue<>* a_queue) {
        a_queue->qfn();
    }


#ifdef __APPLE__
#pragma mark -
#pragma mark symbol
#endif

    // parts of the symbol class but must be defined after atom is defined

    symbol::symbol(const atom& value) {
        s = value;
    }


    symbol& symbol::operator=(const atom& value) {
        s = value;
        return *this;
    }


#ifdef __APPLE__
#pragma mark -
#pragma mark atom
#endif


    bool atom::operator==(const max::t_symbol* s) const {
        return atom_getsym(this) == s;
    }


    bool atom::operator==(const symbol s) const {
        return atom_getsym(this) == (const max::t_symbol*)s;
    }


    bool atom::operator==(const char* str) const {
        return atom_getsym(this) == max::gensym(str);
    }


    bool atom::operator==(const bool value) const {
        return (atom_getlong(this) != 0) == value;
    }


    bool atom::operator==(const int value) const {
        return atom_getlong(this) == value;
    }


    bool atom::operator==(const long value) const {
        return atom_getlong(this) == value;
    }


    bool atom::operator==(const double value) const {
        return atom_getfloat(this) == value;
    }


    bool atom::operator==(const max::t_object* value) const {
        return atom_getobj(this) == value;
    }


    bool atom::operator==(const max::t_atom& b) const {
        return this->a_type == b.a_type && this->a_w.w_obj == b.a_w.w_obj;
    }


    bool atom::operator==(const time_value value) const {
        const max::t_atom& a = *this;
        return atom_getfloat(&a) == static_cast<double>(value);
    }


#ifdef __APPLE__
#pragma mark -
#pragma mark outlet_call_is_safe
#endif


    // specialized implementations of outlet_call_is_safe() used by outlet<> implementation

    template<>
    bool outlet_call_is_safe<thread_check::main>() {
        if (max::systhread_ismainthread())
            return true;
        else
            return false;
    };


    template<>
    bool outlet_call_is_safe<thread_check::scheduler>() {
        if (max::systhread_istimerthread())
            return true;
        else
            return false;
    };


    template<>
    bool outlet_call_is_safe<thread_check::any>() {
        if (max::systhread_ismainthread() || max::systhread_istimerthread())
            return true;
        else
            return false;
    };


    template<>
    bool outlet_call_is_safe<thread_check::none>() {
        return true;
    };


#ifdef __APPLE__
#pragma mark -
#pragma mark vector_operator
#endif


    // implementation of sample_operator-style calls made to a vector_operator

    template<placeholder vector_operator_placeholder_type>
    sample vector_operator<vector_operator_placeholder_type>::operator()(const sample x) {
        sample        input_storage[1]   { x };
        sample        output_storage[1]  {};
        sample*       input              { input_storage };
        sample*       output             { output_storage };
        audio_bundle  input_bundle       { &input, 1, 1 };
        audio_bundle  output_bundle      { &output, 1, 1 };

        (*this)(input_bundle, output_bundle);

        return output[0];
    }


#ifdef __APPLE__
#pragma mark -
#pragma mark message<>
#endif


    void deferred_message::pop() {
        deferred_message x;
        if (m_owning_message->m_deferred_messages.try_dequeue(x))
            x.m_owning_message->m_function(x.m_args, x.m_inlet);
    }


}    // namespace c74::min
