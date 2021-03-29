/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

    struct t_object;

    /** The symbol.

        Note: You should <em>never</em> manipulate the s_name field of the #t_symbol directly!
        Doing so will corrupt Max's symbol table.
        Instead, <em>always</em> use gensym() to get a symbol with the desired string
        contents for the s_name field.

        @ingroup symbol
     */
    struct t_symbol {
        const char*		s_name;		///< name: a c-string
        t_object*		s_thing;	///< possible binding to a t_object
    };


    BEGIN_USING_C_LINKAGE


    /**
        Generates a unique #t_symbol* . The symbol will be formatted somewhat like "u123456789".

        @ingroup	misc
        @return 	This function returns a unique #t_symbol* .
     */
    t_symbol* symbol_unique(void);


    /**
        Strip quotes from the beginning and end of a symbol if they are present.
        @ingroup	misc
        @param	s	The symbol to be stipped.
        @return		Symbol with any leading/trailing quote pairs removed.
     */
    t_symbol* symbol_stripquotes(t_symbol* s);





    END_USING_C_LINKAGE

}} // namespace c74::max
