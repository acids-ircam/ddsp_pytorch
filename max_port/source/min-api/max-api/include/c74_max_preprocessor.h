/*
    Copyright (c) 2011, Stefan Reinalter

    This software is provided 'as-is', without any express or implied
    warranty. In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

    3. This notice may not be removed or altered from any source
    distribution.
*/

// ext_preprocessor.h:
// - derived from preprocessor magic here: http://altdevblogaday.com/wp-content/uploads/2011/10/Assert.txt
// - macros changed from ME_PP_* to C74_* to avoid any name conflicts

#pragma once

/// concatenates tokens, even when the tokens are macros themselves
#define C74_JOIN_HELPER_HELPER(_0, _1)		_0##_1
#define C74_JOIN_HELPER(_0, _1)				C74_JOIN_HELPER_HELPER(_0, _1)
#define C74_JOIN_IMPL(_0, _1)					C74_JOIN_HELPER(_0, _1)

#define C74_JOIN_2(_0, _1)																	C74_JOIN_IMPL(_0, _1)
#define C74_JOIN_3(_0, _1, _2)																C74_JOIN_2(C74_JOIN_2(_0, _1), _2)
#define C74_JOIN_4(_0, _1, _2, _3)															C74_JOIN_2(C74_JOIN_3(_0, _1, _2), _3)
#define C74_JOIN_5(_0, _1, _2, _3, _4)														C74_JOIN_2(C74_JOIN_4(_0, _1, _2, _3), _4)
#define C74_JOIN_6(_0, _1, _2, _3, _4, _5)													C74_JOIN_2(C74_JOIN_5(_0, _1, _2, _3, _4), _5)
#define C74_JOIN_7(_0, _1, _2, _3, _4, _5, _6)												C74_JOIN_2(C74_JOIN_6(_0, _1, _2, _3, _4, _5), _6)
#define C74_JOIN_8(_0, _1, _2, _3, _4, _5, _6, _7)											C74_JOIN_2(C74_JOIN_7(_0, _1, _2, _3, _4, _5, _6), _7)
#define C74_JOIN_9(_0, _1, _2, _3, _4, _5, _6, _7, _8)										C74_JOIN_2(C74_JOIN_8(_0, _1, _2, _3, _4, _5, _6, _7), _8)
#define C74_JOIN_10(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9)									C74_JOIN_2(C74_JOIN_9(_0, _1, _2, _3, _4, _5, _6, _7, _8), _9)
#define C74_JOIN_11(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10)							C74_JOIN_2(C74_JOIN_10(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9), _10)
#define C74_JOIN_12(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11)						C74_JOIN_2(C74_JOIN_11(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10), _11)
#define C74_JOIN_13(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12)					C74_JOIN_2(C74_JOIN_12(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11), _12)
#define C74_JOIN_14(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13)				C74_JOIN_2(C74_JOIN_13(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12), _13)
#define C74_JOIN_15(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14)		C74_JOIN_2(C74_JOIN_14(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13), _14)
#define C74_JOIN_16(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15)	C74_JOIN_2(C74_JOIN_15(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14), _15)


/// chooses a value based on a condition
#define C74_IF_0(t, f)			f
#define C74_IF_1(t, f)			t
#define C74_IF(cond, t, f)		C74_JOIN_2(C74_IF_, C74_TO_BOOL(cond))(t, f)


/// converts a condition into a boolean 0 (=false) or 1 (=true)
#define C74_TO_BOOL_0 0
#define C74_TO_BOOL_1 1
#define C74_TO_BOOL_2 1
#define C74_TO_BOOL_3 1
#define C74_TO_BOOL_4 1
#define C74_TO_BOOL_5 1
#define C74_TO_BOOL_6 1
#define C74_TO_BOOL_7 1
#define C74_TO_BOOL_8 1
#define C74_TO_BOOL_9 1
#define C74_TO_BOOL_10 1
#define C74_TO_BOOL_11 1
#define C74_TO_BOOL_12 1
#define C74_TO_BOOL_13 1
#define C74_TO_BOOL_14 1
#define C74_TO_BOOL_15 1
#define C74_TO_BOOL_16 1

#define C74_TO_BOOL(x)		C74_JOIN_2(C74_TO_BOOL_, x)


/// Returns 1 if the arguments to the variadic macro are separated by a comma, 0 otherwise.
#define C74_HAS_COMMA(...)							C74_HAS_COMMA_EVAL(C74_HAS_COMMA_ARGS(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0))
#define C74_HAS_COMMA_EVAL(...)						__VA_ARGS__
#define C74_HAS_COMMA_ARGS(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, ...) _16


/// Returns 1 if the argument list to the variadic macro is empty, 0 otherwise.
#define C74_IS_EMPTY(...)														\
    C74_HAS_COMMA																\
    (																			\
        C74_JOIN_5                                                              \
        (																		\
            C74_IS_EMPTY_CASE_,                                                 \
            C74_HAS_COMMA(__VA_ARGS__),                                         \
            C74_HAS_COMMA(C74_IS_EMPTY_BRACKET_TEST __VA_ARGS__),               \
            C74_HAS_COMMA(__VA_ARGS__ (~)),                                     \
            C74_HAS_COMMA(C74_IS_EMPTY_BRACKET_TEST __VA_ARGS__ (~))            \
        )																		\
    )

#define C74_IS_EMPTY_CASE_0001			,
#define C74_IS_EMPTY_BRACKET_TEST(...)	,


// C74_VA_NUM_ARGS() is a very nifty macro to retrieve the number of arguments handed to a variable-argument macro.
// unfortunately, VS 2010 still has this preprocessor bug which treats a __VA_ARGS__ argument as being one single parameter:
// https://connect.microsoft.com/VisualStudio/feedback/details/521844/variadic-macro-treating-va-args-as-a-single-parameter-for-other-macros#details
#if _MSC_VER >= 1400
#	define C74_VA_NUM_ARGS_HELPER(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, N, ...)	N
#	define C74_VA_NUM_ARGS_REVERSE_SEQUENCE			16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
#	define C74_VA_NUM_ARGS_LEFT (
#	define C74_VA_NUM_ARGS_RIGHT )
#	define C74_VA_NUM_ARGS(...)						C74_VA_NUM_ARGS_HELPER C74_VA_NUM_ARGS_LEFT __VA_ARGS__, C74_VA_NUM_ARGS_REVERSE_SEQUENCE C74_VA_NUM_ARGS_RIGHT
#else
#	define C74_VA_NUM_ARGS(...)						C74_VA_NUM_ARGS_HELPER(__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#	define C74_VA_NUM_ARGS_HELPER(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, N, ...)	N
#endif

// C74_NUM_ARGS correctly handles the case of 0 arguments
#define C74_NUM_ARGS(...)								C74_IF(C74_IS_EMPTY(__VA_ARGS__), 0, C74_VA_NUM_ARGS(__VA_ARGS__))


// C74_PASS_ARGS passes __VA_ARGS__ as multiple parameters to another macro, working around the following bug:
// https://connect.microsoft.com/VisualStudio/feedback/details/521844/variadic-macro-treating-va-args-as-a-single-parameter-for-other-macros#details
#if _MSC_VER >= 1400
#	define C74_PASS_ARGS_LEFT (
#	define C74_PASS_ARGS_RIGHT )
#	define C74_PASS_ARGS(...)							C74_PASS_ARGS_LEFT __VA_ARGS__ C74_PASS_ARGS_RIGHT
#else
#	define C74_PASS_ARGS(...)							(__VA_ARGS__)
#endif


/// Expand any number of arguments into a list of operations called with those arguments
#define C74_EXPAND_ARGS_0(op, empty)
#define C74_EXPAND_ARGS_1(op, a1)																			op(a1, 0)
#define C74_EXPAND_ARGS_2(op, a1, a2)																		op(a1, 0) op(a2, 1)
#define C74_EXPAND_ARGS_3(op, a1, a2, a3)																	op(a1, 0) op(a2, 1) op(a3, 2)
#define C74_EXPAND_ARGS_4(op, a1, a2, a3, a4)																op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3)
#define C74_EXPAND_ARGS_5(op, a1, a2, a3, a4, a5)															op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4)
#define C74_EXPAND_ARGS_6(op, a1, a2, a3, a4, a5, a6)														op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5)
#define C74_EXPAND_ARGS_7(op, a1, a2, a3, a4, a5, a6, a7)													op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6)
#define C74_EXPAND_ARGS_8(op, a1, a2, a3, a4, a5, a6, a7, a8)												op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7)
#define C74_EXPAND_ARGS_9(op, a1, a2, a3, a4, a5, a6, a7, a8, a9)											op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8)
#define C74_EXPAND_ARGS_10(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)									op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9)
#define C74_EXPAND_ARGS_11(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)								op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10)
#define C74_EXPAND_ARGS_12(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)							op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10) op(a12, 11)
#define C74_EXPAND_ARGS_13(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)					op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10) op(a12, 11) op(a13, 12)
#define C74_EXPAND_ARGS_14(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)				op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10) op(a12, 11) op(a13, 12) op(a14, 13)
#define C74_EXPAND_ARGS_15(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)			op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10) op(a12, 11) op(a13, 12) op(a14, 13) op(a15, 14)
#define C74_EXPAND_ARGS_16(op, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16)		op(a1, 0) op(a2, 1) op(a3, 2) op(a4, 3) op(a5, 4) op(a6, 5) op(a7, 6) op(a8, 7) op(a9, 8) op(a10, 9) op(a11, 10) op(a12, 11) op(a13, 12) op(a14, 13) op(a15, 14) op(a16, 15)

#define C74_EXPAND_ARGS(op, ...)		C74_JOIN_2(C74_EXPAND_ARGS_, C74_NUM_ARGS(__VA_ARGS__)) C74_PASS_ARGS(op, __VA_ARGS__)


/// Turns any legal C++ expression into nothing
#define C74_UNUSED_IMPL(symExpr, n)					, (void)sizeof(symExpr)
#define C74_UNUSED(...)								(void)sizeof(true) C74_EXPAND_ARGS C74_PASS_ARGS(C74_UNUSED_IMPL, __VA_ARGS__)

#define C74_VARFUN_0(VARFUN_IMPL)  VARFUN_IMPL((void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0)
#define C74_VARFUN_1(VARFUN_IMPL, p1)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0)
#define C74_VARFUN_2(VARFUN_IMPL, p1, p2)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)(p2), (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0)
#define C74_VARFUN_3(VARFUN_IMPL, p1, p2, p3)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)(p2), (void*)(c74::max::t_atom_long)(p3), (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0)
#define C74_VARFUN_4(VARFUN_IMPL, p1, p2, p3, p4)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)(p2), (void*)(c74::max::t_atom_long)(p3), (void*)(c74::max::t_atom_long)(p4), (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0)
#define C74_VARFUN_5(VARFUN_IMPL, p1, p2, p3, p4, p5)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)(p2), (void*)(c74::max::t_atom_long)(p3), (void*)(c74::max::t_atom_long)(p4), (void*)(c74::max::t_atom_long)(p5), (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0)
#define C74_VARFUN_6(VARFUN_IMPL, p1, p2, p3, p4, p5, p6)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)(p2), (void*)(c74::max::t_atom_long)(p3), (void*)(c74::max::t_atom_long)(p4), (void*)(c74::max::t_atom_long)(p5), (void*)(c74::max::t_atom_long)(p6), (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0)
#define C74_VARFUN_7(VARFUN_IMPL, p1, p2, p3, p4, p5, p6, p7)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)(p2), (void*)(c74::max::t_atom_long)(p3), (void*)(c74::max::t_atom_long)(p4), (void*)(c74::max::t_atom_long)(p5), (void*)(c74::max::t_atom_long)(p6), (void*)(c74::max::t_atom_long)(p7), (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0)
#define C74_VARFUN_8(VARFUN_IMPL, p1, p2, p3, p4, p5, p6, p7, p8)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)(p2), (void*)(c74::max::t_atom_long)(p3), (void*)(c74::max::t_atom_long)(p4), (void*)(c74::max::t_atom_long)(p5), (void*)(c74::max::t_atom_long)(p6), (void*)(c74::max::t_atom_long)(p7), (void*)(c74::max::t_atom_long)(p8), (void*)(c74::max::t_atom_long)0, (void*)(c74::max::t_atom_long)0)
#define C74_VARFUN_9(VARFUN_IMPL, p1, p2, p3, p4, p5, p6, p7, p8, p9)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)(p2), (void*)(c74::max::t_atom_long)(p3), (void*)(c74::max::t_atom_long)(p4), (void*)(c74::max::t_atom_long)(p5), (void*)(c74::max::t_atom_long)(p6), (void*)(c74::max::t_atom_long)(p7), (void*)(c74::max::t_atom_long)(p8), (void*)(c74::max::t_atom_long)(p9), (void*)(c74::max::t_atom_long)0)
#define C74_VARFUN_10(VARFUN_IMPL, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)(p2), (void*)(c74::max::t_atom_long)(p3), (void*)(c74::max::t_atom_long)(p4), (void*)(c74::max::t_atom_long)(p5), (void*)(c74::max::t_atom_long)(p6), (void*)(c74::max::t_atom_long)(p7), (void*)(c74::max::t_atom_long)(p8), (void*)(c74::max::t_atom_long)(p9), (void*)(c74::max::t_atom_long)(p10))
#define C74_VARFUN_11(VARFUN_IMPL, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11)  VARFUN_IMPL((void*)(c74::max::t_atom_long)(p1), (void*)(c74::max::t_atom_long)(p2), (void*)(c74::max::t_atom_long)(p3), (void*)(c74::max::t_atom_long)(p4), (void*)(c74::max::t_atom_long)(p5), (void*)(c74::max::t_atom_long)(p6), (void*)(c74::max::t_atom_long)(p7), (void*)(c74::max::t_atom_long)(p8), (void*)(c74::max::t_atom_long)(p9), (void*)(c74::max::t_atom_long)(p10), (void*)(c74::max::t_atom_long)(p11))

// C74_VARFUN supports calling a function via a macro in a way that looks like a var args function
//  - VARFUN_IMPL: is a function that takes 10 void* parameters
//  - there must be at least one parameter passed after the VARFUN_IMPL
//  - each parameter passed is cast first to a c74::max::t_atom_long and then to a void*
//    this is done to sign-extend and promote shorter integer types
//  - see object_method for an example

#if _MSC_VER >= 1400
#define C74_VARFUN(VARFUN_IMPL, ...) C74_JOIN_2(C74_VARFUN_, C74_NUM_ARGS(__VA_ARGS__)) C74_PASS_ARGS(VARFUN_IMPL, __VA_ARGS__)
#else
#define C74_VARFUN(VARFUN_IMPL, ...) C74_JOIN_2(C74_VARFUN_, C74_NUM_ARGS(__VA_ARGS__))( VARFUN_IMPL, __VA_ARGS__ )
#endif
