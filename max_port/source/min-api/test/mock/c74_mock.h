/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <iostream>
#include <mutex>
#include <ratio>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>


// Mac or Win detection
#if !defined(WIN_VERSION) && !defined(MAC_VERSION)
    #ifdef __APPLE__
        #define MAC_VERSION
    #else
        #define WIN_VERSION
    #endif
#endif  // #if !defined(MAC_VERSION) && !defined(WIN_VERSION)

// 64-bit detection
#ifdef WIN_VERSION
    #ifdef _WIN64
        #define C74_X64
    #endif
#endif  // #ifdef WIN_VERSION

#ifdef MAC_VERSION
    #if __LP64__
        #define C74_X64
    #endif
#endif // #ifdef MAC_VERSION

#ifdef WIN_VERSION
    #ifndef snprintf
        #define snprintf _snprintf
    #endif

    #define MOCK_EXPORT extern "C" __declspec(dllexport)
    #define MOCK_EXPORT_CPP __declspec(dllexport)
#else
    #define MOCK_EXPORT extern "C"
    #define MOCK_EXPORT_CPP
#endif


// Max

#include "c74_mock_atoms.h"
#include "c74_mock_classes.h"
#include "c74_mock_inlets.h"
#include "c74_mock_outlets.h"
#include "c74_mock_kernel.h"
#include "c74_mock_memory.h"
#include "c74_mock_misc.h"
#include "c74_mock_dictionary.h"
#include "c74_mock_clock.h"
#include "c74_mock_ui.h"

// MSP
#include "c74_mock_msp.h"

// Jitter
//#include "c74_mock_jitter.h"


    namespace c74 {
    namespace max {


    /** Send int or float or symbol messages to an object.
        Sort of like object_method(), but much simpler with no dependencies etc.
        Unlike object_method() you can specify the inlet for which the message is intended.

        @param x		The object to which the message is sent.
        @param value	The int, float, or symbol to send.
        @param inletnum	Optional inlet number for the message.  Defaults to inlet 0.
        @return			Whatever value is returned from the method.
    */
    template <class T>
    void *mock_send_message(void *x, T value, int inletnum=0) {
        t_object		*o = (t_object*)x;
        object_inlets*	inlets = (object_inlets*)o->o_inlet;
        t_mock_inlet&	inlet = inlets->mock_inlets[inletnum];

        return inlet.push(value);
    }


}} // namespace c74::max
