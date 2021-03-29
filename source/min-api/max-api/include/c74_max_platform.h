/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cfloat>

#include <algorithm>


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


// These macros are used to ensure that Win32 builds of Max externals use a 2 byte struct packing.
// On Mac and Win64 we use the default byte packing.
#if ( defined(WIN_VERSION) && !defined(C74_X64) && !defined(TWOBLUECUBES_CATCH_HPP_INCLUDED))
    #define C74_PRAGMA_STRUCT_PACKPUSH	1
#else // MAC_VERSION or Win64
    #define C74_PRAGMA_STRUCT_PACKPUSH	0
#endif


// Types
namespace c74 {
namespace max {
    typedef int 				t_int;
    typedef unsigned int 		t_uint; 	///< an unsigned int as defined by the architecture / platform  @ingroup misc
    typedef int8_t 				t_int8; 	///< a 1-byte int  @ingroup misc
    typedef uint8_t 			t_uint8;	///< an unsigned 1-byte int  @ingroup misc
    typedef int16_t 			t_int16; 	///< a 2-byte int  @ingroup misc
    typedef uint16_t	 		t_uint16; 	///< an unsigned 2-byte int  @ingroup misc
    typedef int32_t 			t_int32; 	///< a 4-byte int  @ingroup misc
    typedef uint32_t 			t_uint32; 	///< an unsigned 4-byte int  @ingroup misc
    typedef int64_t 			t_int64;	///< an 8-byte int  @ingroup misc
    typedef uint64_t 			t_uint64;	///< an unsigned 8-byte int  @ingroup misc
    typedef t_uint32 			t_fourcc; 	///< an integer of suitable size to hold a four char code / identifier  @ingroup misc

    #ifdef C74_X64
    typedef uint64_t			t_ptr_uint;		///< an unsigned pointer-sized int  @ingroup misc
    typedef int64_t				t_ptr_int; 		///< a pointer-sized int  @ingroup misc
    typedef double				t_atom_float;	///< the type that is an A_FLOAT in a #t_atom  @ingroup misc
    #else
    typedef uint32_t			t_ptr_uint;		///< an unsigned pointer-sized int  @ingroup misc
    typedef int64_t 			t_ptr_int; 		///< a pointer-sized int  @ingroup misc
    typedef float 				t_atom_float; 	///< the type that is an A_FLOAT in a #t_atom  @ingroup misc
    #endif

    typedef t_ptr_uint 	t_ptr_size;		///< unsigned pointer-sized value for counting (like size_t)  @ingroup misc
    typedef t_ptr_int 	t_atom_long;	///< the type that is an A_LONG in a #t_atom  @ingroup misc
    typedef t_atom_long	t_max_err;		///< an integer value suitable to be returned as an error code  @ingroup misc

    typedef char** t_handle;			///< a handle (address of a pointer)  @ingroup misc
    typedef char* t_ptr;				///< a pointer  @ingroup misc

    typedef t_uint8 t_bool; 			///< a true/false variable  @ingroup misc
    typedef t_int16 t_filepath;			///< i.e. path/vol in file APIs identifying a folder  @ingroup misc
}} // namespace c74::max

// Misc

#ifdef WIN_VERSION
    #define C74_EXPORT __declspec(dllexport)
    #define C74_MUST_CHECK

    // promote this warning to an error because
    // it helps catch using long* when we really need t_atom_long* for x64
    #pragma warning ( error : 4133 ) // incompatible types

    //#ifndef _CRT_SECURE_NO_WARNINGS
    //#define _CRT_SECURE_NO_WARNINGS
    //#endif
    //
    //#ifndef _SCL_SECURE_NO_WARNINGS
    //#define _SCL_SECURE_NO_WARNINGS
    //#endif

    #define NOMINMAX
    #include "windows.h"

    #define strncpy(dst, src, size) strncpy_s(dst, size, src, _TRUNCATE);
    #define snprintf(dst, size, fmt, ...) _snprintf_s(dst, size, _TRUNCATE, fmt, __VA_ARGS__)

#else // MAC_VERSION
    #define C74_EXPORT __attribute__((visibility("default")))
    /** If the result of a function is unused, force a compiler warning about it. */
    #define C74_MUST_CHECK __attribute__((warn_unused_result))
#endif


#ifdef C74_NO_DEPRECATION
    #define C74_DEPRECATED(func) func
#endif

#ifndef C74_DEPRECATED
    #ifdef __GNUC__
        #define C74_DEPRECATED(func) func __attribute__ ((deprecated))
    #elif defined(_MSC_VER)
        #define C74_DEPRECATED(func) __declspec(deprecated) func
    #else
        #define C74_DEPRECATED(func) func
    #endif
#endif // C74_DEPRECATED





// Atomics
#ifdef MAC_VERSION
#include <libkern/OSAtomic.h>
#endif

namespace c74 {
namespace max {


    #ifdef MAC_VERSION

    typedef volatile int32_t t_int32_atomic;
    typedef volatile int64_t t_int64_atomic;
    typedef volatile u_int32_t t_uint32_atomic;
    typedef volatile u_int64_t t_uint64_atomic;

    /** increment an atomic int value
        @ingroup threading
        return value of ATOMIC_INCREMENT and ATOMIC_DECREMENT is the *new* value after performing the operation
    */
    #define ATOMIC_INCREMENT(atomicptr) OSAtomicIncrement32((int32_t*)atomicptr)

    /** increment an atomic int value with a memory barrier
        @ingroup threading
        return value of ATOMIC_INCREMENT and ATOMIC_DECREMENT is the *new* value after performing the operation
    */
    #define ATOMIC_INCREMENT_BARRIER(atomicptr) OSAtomicIncrement32Barrier((int32_t*)atomicptr)

    /** decrement an atomic int value
        @ingroup threading
        return value of ATOMIC_INCREMENT and ATOMIC_DECREMENT is the *new* value after performing the operation
    */
    #define ATOMIC_DECREMENT(atomicptr) OSAtomicDecrement32((int32_t*)atomicptr)

    /** decrement an atomic int value with a memory barrier
        @ingroup threading
        return value of ATOMIC_INCREMENT and ATOMIC_DECREMENT is the *new* value after performing the operation
    */
    #define ATOMIC_DECREMENT_BARRIER(atomicptr) OSAtomicDecrement32Barrier((int32_t*)atomicptr)

    #define ATOMIC_COMPARE_SWAP32(oldvalue, newvalue, atomicptr) (OSAtomicCompareAndSwap32Barrier(oldvalue, newvalue, atomicptr))
    #define ATOMIC_COMPARE_SWAP64(oldvalue, newvalue, atomicptr) (OSAtomicCompareAndSwap64Barrier(oldvalue, newvalue, atomicptr))

    #else // WIN_VERSION

    // rbs: intrin.h is not compatible with C, only C++
    // #include <intrin.h>
    typedef volatile long t_int32_atomic;
    typedef volatile __int64 t_int64_atomic;
    typedef volatile unsigned long t_uint32_atomic;
    typedef volatile unsigned __int64 t_uint64_atomic;

    #pragma intrinsic (_InterlockedIncrement)
    #pragma intrinsic (_InterlockedDecrement)
    #pragma intrinsic (_InterlockedCompareExchange)
    #pragma intrinsic (_InterlockedCompareExchange64)

    /**	Use this routine for incrementing a global counter using a threadsafe and multiprocessor safe method.
        @ingroup			threading
        @param	atomicptr	pointer to the (int) counter.
    */

    // on windows I don't think there are non-barrier atomic increment / decrement functions
    // perhaps could be done with inline assembly?

    #define ATOMIC_INCREMENT(atomicptr)			  (_InterlockedIncrement(atomicptr))
    #define ATOMIC_INCREMENT_BARRIER(atomicptr)   (_InterlockedIncrement(atomicptr))


    /**	Use this routine for decrementing a global counter using a threadsafe and multiprocessor safe method.
        @ingroup	threading
        @param	atomicptr	pointer to the (int) counter.
    */
    #define ATOMIC_DECREMENT(atomicptr)			  (_InterlockedDecrement(atomicptr))
    #define ATOMIC_DECREMENT_BARRIER(atomicptr)   (_InterlockedDecrement(atomicptr))

    /** atomic compare exchange does this:
        - if (*atomicptr == oldvalue) *atomicptr = newvalue;
        - all of above done atomically
        - return value is boolean: true if exchange was done
        @ingroup	threading
        @param	atomicptr		pointer to the atomic value
        @param	newvalue		value that will be assigned to *atomicptr if test succeeds
        @param	oldvalue		newvalue is only stored if original value equals oldvalue
    */
    #define ATOMIC_COMPARE_SWAP32(oldvalue, newvalue, atomicptr) (_InterlockedCompareExchange(atomicptr, newvalue, oldvalue)==oldvalue)
    #define ATOMIC_COMPARE_SWAP64(oldvalue, newvalue, atomicptr) (_InterlockedCompareExchange64(atomicptr, newvalue, oldvalue)==oldvalue)

    #endif // WIN_VERSION

}} // namespace c74::max



// C++ Specific

#ifdef __cplusplus
    /**
        Ensure that any definitions following this macro use a C-linkage, not a C++ linkage.
        The Max API uses C-linkage.
        This is important for objects written in C++ or that use a C++ compiler.
        This macro must be balanced with the #END_USING_C_LINKAGE macro.
        @ingroup misc
    */
    #define BEGIN_USING_C_LINKAGE \
    extern "C" {
#else
    #error A C++ compiler is required for the use of this API.
    #define BEGIN_USING_C_LINKAGE
#endif // __cplusplus

#ifdef __cplusplus
    /**
        Close a definition section that was opened using #BEGIN_USING_C_LINKAGE.
        @ingroup misc
    */
    #define END_USING_C_LINKAGE \
        }
#else
    #define END_USING_C_LINKAGE
#endif // __cplusplus
