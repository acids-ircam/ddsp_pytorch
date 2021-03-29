/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

    /// officially recognized t_atom types @ingroup atom
    enum e_max_atomtypes : long {
        A_NOTHING = 0,	///< no type, thus no atom
        A_LONG,			///< integer (32-bit on 32-bit arch, 64-bit on 64-bit arch)
        A_FLOAT,		///< decimal (float on 32-bit arch, double on 64-bit arch)
        A_SYM,			///< symbol
        A_OBJ,			///< object
        A_DEFLONG,		///< long but defaults to zero
        A_DEFFLOAT,		///< float, but defaults to zero
        A_DEFSYM,		///< symbol, defaults to ""
        A_GIMME,		///< request that args be passed as an array, the routine will check the types itself.
        A_CANT,			///< cannot typecheck args
        A_SEMI,			///< semicolon
        A_COMMA,		///< comma
        A_DOLLAR,		///< dollar
        A_DOLLSYM,		///< dollar
        A_GIMMEBACK,	///< request that args be passed as an array, the routine will check the types itself. can return atom value in final atom ptr arg. function returns long error code 0 = no err. see gimmeback_meth typedef

        A_DEFER	=		0x41,	///< A special signature for declaring methods. This is like A_GIMME, but the call is deferred.
        A_USURP =		0x42,	///< A special signature for declaring methods. This is like A_GIMME, but the call is deferred and multiple calls within one servicing of the queue are filtered down to one call.
        A_DEFER_LOW =	0x43,	///< A special signature for declaring methods. This is like A_GIMME, but the call is deferref to the back of the queue.
        A_USURP_LOW =	0x44	///< A special signature for declaring methods. This is like A_GIMME, but the call is deferred to the back of the queue and multiple calls within one servicing of the queue are filtered down to one call.

    };

    /// Union for packing any of the datum defined in #e_max_atomtypes. @ingroup atom
    union word {
        t_atom_long		w_long;		///< long integer
        t_atom_float	w_float;	///< 32-bit float
        t_symbol*		w_sym;		///< pointer to a symbol in the Max symbol table
        t_object*		w_obj;		///< pointer to a #t_object or other generic pointer
    };

    /// An atom is a typed datum. @ingroup atom
    struct t_atom {
        short			a_type;
        union word		a_w;
    };

    /// Flags that determine how functions convert atoms into text (C-strings). @ingroup atom
    enum e_max_atom_gettext_flags {
        OBEX_UTIL_ATOM_GETTEXT_DEFAULT =			0x00000000, ///< default translation rules for getting text from atoms
        OBEX_UTIL_ATOM_GETTEXT_TRUNCATE_ZEROS =		0x00000001, ///< eliminate redundant zeros for floating point numbers (default used)
        OBEX_UTIL_ATOM_GETTEXT_SYM_NO_QUOTE	=		0x00000002, ///< don't introduce quotes around symbols with spaces
        OBEX_UTIL_ATOM_GETTEXT_SYM_FORCE_QUOTE =	0x00000004, ///< always introduce quotes around symbols (useful for JSON)
        OBEX_UTIL_ATOM_GETTEXT_COMMA_DELIM =		0x00000008, ///< separate atoms with commas (useful for JSON)
        OBEX_UTIL_ATOM_GETTEXT_FORCE_ZEROS =		0x00000010, ///< always print the zeros
        OBEX_UTIL_ATOM_GETTEXT_NUM_HI_RES =			0x00000020,	///< print more decimal places
        OBEX_UTIL_ATOM_GETTEXT_NUM_LO_RES =			0x00000040  ///< // print fewer decimal places (HI_RES will win though)
    };

    BEGIN_USING_C_LINKAGE

    /// Inserts an integer into a #t_atom and change the t_atom's type to #A_LONG.
    /// @ingroup atom
    /// @param 	a		Pointer to a #t_atom whose value and type will be modified
    /// @param 	b		Integer value to copy into the #t_atom
    /// @return 		This function returns the error code #MAX_ERR_NONE if successful,
    ///  				or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    t_max_err atom_setlong(t_atom* a, t_atom_long b);

    ///	Inserts a floating point number into a #t_atom and change the t_atom's type to #A_FLOAT.
    ///	@ingroup atom
    ///	@param 	a		Pointer to a #t_atom whose value and type will be modified
    ///	@param 	b		Floating point value to copy into the #t_atom
    ///	@return 		This function returns the error code #MAX_ERR_NONE if successful,
    ///	 				or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    t_max_err atom_setfloat(t_atom* a, double b);

    ///	Inserts a #t_symbol*  into a #t_atom and change the t_atom's type to #A_SYM.
    ///	@ingroup atom
    ///	@param 	a		Pointer to a #t_atom whose value and type will be modified
    ///	@param 	b		Pointer to a #t_symbol to copy into the #t_atom
    ///	@return 		This function returns the error code #MAX_ERR_NONE if successful,
    ///	 				or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    t_max_err atom_setsym(t_atom* a, const t_symbol* b);

    ///	Inserts a generic pointer value into a #t_atom and change the t_atom's type to #A_OBJ.
    ///	@ingroup atom
    ///	@param 	a		Pointer to a #t_atom whose value and type will be modified
    ///	@param 	b		Pointer value to copy into the #t_atom
    ///	@return 		This function returns the error code #MAX_ERR_NONE if successful,
    ///	 				or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    t_max_err atom_setobj(t_atom* a, void* b);

    ///	Retrieves a long integer value from a #t_atom.
    ///	@ingroup atom
    ///	@param 	a		Pointer to a #t_atom whose value is of interest
    ///	@return 		This function returns the value of the specified #t_atom as an integer, if possible. Otherwise, it returns 0.
    ///	@remark 		If the #t_atom is not of the type specified by the function, the function will attempt to coerce a valid value from the t_atom.
    ///					For instance, if the t_atom <tt>at</tt> is set to type #A_FLOAT with a value of <tt>3.7</tt>,
    ///					the atom_getlong() function will return the truncated integer value of <tt>at</tt>, or <tt>3</tt>.
    ///					An attempt is also made to coerce #t_symbol data.
    t_atom_long atom_getlong(const t_atom* a);

    ///	Retrieves a floating point value from a #t_atom.
    ///	@ingroup atom
    ///	@param 	a		Pointer to a #t_atom whose value is of interest
    ///	@return 		This function returns the value of the specified #t_atom as a floating point number, if possible. Otherwise, it returns 0.
    ///	@remark 		If the #t_atom is not of the type specified by the function, the function will attempt to coerce a valid value from the t_atom.
    ///					For instance, if the t_atom <tt>at</tt> is set to type #A_LONG with a value of <tt>5</tt>,
    ///					the atom_getfloat() function will return the value of <tt>at</tt> as a float, or <tt>5.0</tt>.
    ///					An attempt is also made to coerce #t_symbol data.
    t_atom_float atom_getfloat(const t_atom* a);

    ///	Retrieves a t_symbol*  value from a t_atom.
    ///	@ingroup atom
    ///	@param 	a		Pointer to a t_atom whose value is of interest
    ///	@return 		This function returns the value of the specified #A_SYM-typed #t_atom, if possible.
    ///					Otherwise, it returns an empty, but valid, #t_symbol* , equivalent to <tt>gensym("")</tt>, or <tt>_sym_nothing</tt>.
    ///
    ///	@remark 		No attempt is made to coerce non-matching data types.
    t_symbol* atom_getsym(const t_atom* a);

    ///	Retrieves a generic pointer value from a #t_atom.
    ///	@ingroup atom
    ///	@param 	a		Pointer to a #t_atom whose value is of interest
    ///	@return 		This function returns the value of the specified #A_OBJ-typed t_atom, if possible. Otherwise, it returns NULL.
    void* atom_getobj(const t_atom* a);

    ///	Retrieves an unsigned integer value between 0 and 255 from a t_atom.
    ///	@ingroup atom
    ///	@param 	a		Pointer to a #t_atom whose value is of interest
    ///	@return 		This function returns the value of the specified #t_atom as an integer between 0 and 255, if possible. Otherwise, it returns 0.
    ///
    ///	@remark 		If the #t_atom is typed #A_LONG, but the data falls outside of the range 0-255, the data is truncated to that range before output.
    ///
    ///	@remark 		If the t_atom is typed #A_FLOAT, the floating point value is multiplied by 255. and truncated to the range 0-255 before output.
    ///					For example, the floating point value <tt>0.5</tt> would be output from atom_getcharfix as <tt>127</tt> (0.5 * 255. = 127.5).
    ///
    ///	@remark 		No attempt is also made to coerce #t_symbol data.
    long atom_getcharfix(const t_atom* a);

    ///	Retrieves type from a #t_atom.
    ///	@ingroup atom
    ///	@param 	a		Pointer to a #t_atom whose type is of interest
    ///	@return 		This function returns the type of the specified t_atom as defined in #e_max_atomtypes
    long atom_gettype(const t_atom* a);

    ///	Parse a C-string into an array of atoms.
    ///	This function allocates memory for the atoms if the ac and av parameters are NULL.
    ///	Otherwise it will attempt to use any memory already allocated to av.
    ///	Any allocated memory should be freed with sysmem_freeptr().
    ///
    ///	@ingroup	atom
    ///	@param		ac			The address of a variable to hold the number of returned atoms.
    ///	@param		av			The address of a #t_atom pointer to which memory may be allocated and atoms copied.
    ///	@param		parsestr	The C-string to parse.
    ///	@return					A Max error code.
    ///
    ///	@remark		The following example will parse the string "foo bar 1 2 3.0" into an array of 5 atoms.
    ///				The atom types will be determined automatically as 2 #A_SYM atoms, 2 #A_LONG atoms, and 1 #A_FLOAT atom.
    ///	@code
    ///	t_atom* av =  NULL;
    ///	long ac = 0;
    ///	t_max_err err = MAX_ERR_NONE;
    ///
    ///	err = atom_setparse(&ac, &av, "foo bar 1 2 3.0");
    ///	@endcode
    t_max_err atom_setparse(long* ac, t_atom** av, const char* parsestr);

    ///	Create an array of atoms populated with values using sprintf-like syntax.
    ///	atom_setformat() supports clfdsoaCLFDSOA tokens
    ///	(primitive type scalars and arrays respectively for the
    ///	char, long, float, double, #t_symbol*, #t_object*, #t_atom*).
    ///	It also supports vbp@ tokens (obval, binbuf, parsestr, attribute).
    ///
    ///	This function allocates memory for the atoms if the ac and av parameters are NULL.
    ///	Otherwise it will attempt to use any memory already allocated to av.
    ///	Any allocated memory should be freed with sysmem_freeptr().
    ///
    ///	@ingroup	atom
    ///	@param		ac			The address of a variable to hold the number of returned atoms.
    ///	@param		av			The address of a #t_atom pointer to which memory may be allocated and atoms copied.
    ///	@param		fmt			An sprintf-style format string specifying values for the atoms.
    ///	@param		...			One or more arguments which are to be substituted into the format string.
    ///	@return		A Max error code.
    ///	@see		atom_getformat()
    ///	@see		atom_setparse()
    t_max_err atom_setformat(long* ac, t_atom** av, const char* fmt, ...);

    ///	Retrieve values from an array of atoms using sscanf-like syntax.
    ///	atom_getformat() supports clfdsoaCLFDSOA tokens
    ///	(primitive type scalars and arrays respectively for the
    ///	char, long, float, double, #t_symbol*, #t_object*, #t_atom*).
    ///	It does not support vbp@ the tokens found in atom_setformat().
    ///	@ingroup	atom
    ///	@param		ac			The number of atoms to parse in av.
    ///	@param		av			The address of the first #t_atom pointer in an array to parse.
    ///	@param		fmt			An sscanf-style format string specifying types for the atoms.
    ///	@param		...			One or more arguments which are address of variables to be set according to the fmt string.
    ///	@return		A Max error code.
    ///	@see		atom_setformat()
    t_max_err atom_getformat(long ac, t_atom* av, const char* fmt, ...);

    ///	Convert an array of atoms into a C-string.
    ///	@ingroup	atom
    ///	@param		ac			The number of atoms to fetch in av.
    ///	@param		av			The address of the first #t_atom pointer in an array to retrieve.
    ///	@param		textsize	The size of the string to which the atoms will be formatted and copied.
    ///	@param		text		The address of the string to which the text will be written.
    ///	@param		flags		Determines the rules by which atoms will be translated into text.
    ///							Values are bit mask as defined by #e_max_atom_gettext_flags.
    ///
    ///	@return		A Max error code.
    ///	@see		atom_setparse()
    t_max_err atom_gettext(long ac, const t_atom* av, long* textsize, char** text, long flags);

    ///	Determines whether or not an atom represents a #t_string object.
    ///	@ingroup	atom
    ///	@param		a				The address of the atom to test.
    ///	@return		Returns true if the #t_atom contains a valid #t_string object.
    long atomisstring(const t_atom* a);

    ///	Determines whether or not an atom represents a #t_atomarray object.
    ///	@ingroup	atom
    ///	@param		a				The address of the atom to test.
    ///	@return		Returns true if the #t_atom contains a valid #t_atomarray object.
    long atomisatomarray(const t_atom* a);

    ///	Determines whether or not an atom represents a #t_dictionary object.
    ///	@ingroup	atom
    ///	@param		a				The address of the atom to test.
    ///	@return		Returns true if the #t_atom contains a valid #t_dictionary object.
    long atomisdictionary(const t_atom* a);

    END_USING_C_LINKAGE

}} // namespace c74::max
