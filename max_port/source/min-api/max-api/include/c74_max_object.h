/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {


    /**
        Find byte offset of a named member of a struct, relative to the beginning of that struct.
        @ingroup misc
        @param	x	The name of the struct
        @param	y	The name of the member
        @return		A pointer-sized integer representing the number of bytes into the struct where the member begins.
    */
    #define calcoffset(x,y) ((c74::max::t_ptr_int)(&(((x *)0L)->y)))



    BEGIN_USING_C_LINKAGE


    // old-school binbuf formatting...
    void binbuf_vinsert(void *x, const char *fmt, ...);
    void binbuf_insert(void *x, t_symbol *ignored, short argc, t_atom *argv);


    // macros for attributes
    // class attributes are almost universally attr_offset, except for class static attributes


    /**
        Create a char attribute and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
    */
    #define CLASS_ATTR_CHAR(c,attrname,flags,structname,structmember) \
        class_addattr((c),attr_offset_new(attrname, c74::max::gensym("char"),(flags),(method)0L,(method)0L,calcoffset(structname,structmember)))


    /**
        Create a t_atom_long integer attribute and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
    */

    #define CLASS_ATTR_LONG(c,attrname,flags,structname,structmember) \
            {		\
                class_addattr((c),attr_offset_new(attrname, c74::max::gensym("long"),(flags),(method)0L,(method)0L,calcoffset(structname,structmember))); \
            }


    #define CLASS_ATTR_FILEPATH(c,attrname,flags,structname,structmember) \
            {		\
                class_addattr((c),attr_offset_new(attrname, c74::max::gensym("filepath"),(flags),(method)0L,(method)0L,calcoffset(structname,structmember))); \
            }

    /**
        Create a 32-bit float attribute and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
    */
    #define CLASS_ATTR_FLOAT(c,attrname,flags,structname,structmember) \
        class_addattr((c),attr_offset_new(attrname, c74::max::gensym("float32"),(flags),(method)0L,(method)0L,calcoffset(structname,structmember)))


    /**
        Create a 64-bit float attribute and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
    */
    #define CLASS_ATTR_DOUBLE(c,attrname,flags,structname,structmember) \
        class_addattr((c),attr_offset_new(attrname, c74::max::gensym("float64"),(flags),(method)0L,(method)0L,calcoffset(structname,structmember)))


    /**
        Create a #t_symbol* attribute and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
    */
    #define CLASS_ATTR_SYM(c,attrname,flags,structname,structmember) \
        class_addattr((c),attr_offset_new(attrname, c74::max::gensym("symbol"),(flags),(method)0L,(method)0L,calcoffset(structname,structmember)))


    /**
        Create a #t_atom attribute and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
    */
    #define CLASS_ATTR_ATOM(c,attrname,flags,structname,structmember) \
        class_addattr((c),attr_offset_new(attrname, c74::max::gensym("atom"),(flags),(method)0L,(method)0L,calcoffset(structname,structmember)))


    /**
        Create a #t_object* attribute and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
    */
    #define CLASS_ATTR_OBJ(c,attrname,flags,structname,structmember) \
        class_addattr((c),attr_offset_new(attrname, c74::max::gensym("object"),(flags),(method)0L,(method)0L,calcoffset(structname,structmember)))




    /**
        Create an array-of-chars attribute of fixed length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	size			The number of chars in the array.
    */
    #define CLASS_ATTR_CHAR_ARRAY(c,attrname,flags,structname,structmember,size) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("char"),(size),(flags),(method)0L,(method)0L,0/*fix*/,calcoffset(structname,structmember)))


    /**
        Create an array-of-long-integers attribute of fixed length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	size			The number of longs in the array.
    */
    #define CLASS_ATTR_LONG_ARRAY(c,attrname,flags,structname,structmember,size) \
            {		\
                class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("long"),(size),(flags),(method)0L,(method)0L,0/*fix*/,calcoffset(structname,structmember))); \
            }

    /**
        Create an array-of-32bit-floats attribute of fixed length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	size			The number of floats in the array.
    */
    #define CLASS_ATTR_FLOAT_ARRAY(c,attrname,flags,structname,structmember,size) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("float32"),(size),(flags),(method)0L,(method)0L,0/*fix*/,calcoffset(structname,structmember)))


    /**
        Create an array-of-64bit-floats attribute of fixed length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	size			The number of doubles in the array.
    */
    #define CLASS_ATTR_DOUBLE_ARRAY(c,attrname,flags,structname,structmember,size) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("float64"),(size),(flags),(method)0L,(method)0L,0/*fix*/,calcoffset(structname,structmember)))


    /**
        Create an array-of-symbols attribute of fixed length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	size			The number of items in the #t_symbol* array.
    */
    #define CLASS_ATTR_SYM_ARRAY(c,attrname,flags,structname,structmember,size) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("symbol"),(size),(flags),(method)0L,(method)0L,0/*fix*/,calcoffset(structname,structmember)))


    /**
        Create an array-of-atoms attribute of fixed length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	size			The number of items in the #t_atom array.
    */
    #define CLASS_ATTR_ATOM_ARRAY(c,attrname,flags,structname,structmember,size) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("atom"),(size),(flags),(method)0L,(method)0L,0/*fix*/,calcoffset(structname,structmember)))


    /**
        Create an array-of-objects attribute of fixed length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	size			The number of items in the #t_object* array.
    */
    #define CLASS_ATTR_OBJ_ARRAY(c,attrname,flags,structname,structmember,size) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("object"),(size),(flags),(method)0L,(method)0L,0/*fix*/,calcoffset(structname,structmember)))




    /**
        Create an array-of-chars attribute of variable length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	sizemember		The actual number of items in the char array at any given moment.
        @param	maxsize			The maximum number of items in the char array, i.e. the number of members allocated for the array in the struct.
    */
    #define CLASS_ATTR_CHAR_VARSIZE(c,attrname,flags,structname,structmember,sizemember,maxsize) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("char"),(maxsize),(flags),(method)0L,(method)0L,calcoffset(structname,sizemember),calcoffset(structname,structmember)))


    /**
        Create an array-of-long-integers attribute of variable length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	sizemember		The actual number of items in the long array at any given moment.
        @param	maxsize			The maximum number of items in the long array, i.e. the number of members allocated for the array in the struct.
    */
    #define CLASS_ATTR_LONG_VARSIZE(c,attrname,flags,structname,structmember,sizemember,maxsize) \
            {		\
                class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("long"),(maxsize),(flags),(method)0L,(method)0L,calcoffset(structname,sizemember),calcoffset(structname,structmember))); \
            }

    /**
        Create an array-of-32bit-floats attribute of variable length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	sizemember		The actual number of items in the float array at any given moment.
        @param	maxsize			The maximum number of items in the float array, i.e. the number of members allocated for the array in the struct.
    */
    #define CLASS_ATTR_FLOAT_VARSIZE(c,attrname,flags,structname,structmember,sizemember,maxsize) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("float32"),(maxsize),(flags),(method)0L,(method)0L,calcoffset(structname,sizemember),calcoffset(structname,structmember)))


    /**
        Create an array-of-64bit-floats attribute of variable length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	sizemember		The actual number of items in the double array at any given moment.
        @param	maxsize			The maximum number of items in the double array, i.e. the number of members allocated for the array in the struct.
    */
    #define CLASS_ATTR_DOUBLE_VARSIZE(c,attrname,flags,structname,structmember,sizemember,maxsize) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("float64"),(maxsize),(flags),(method)0L,(method)0L,calcoffset(structname,sizemember),calcoffset(structname,structmember)))


    /**
        Create an array-of-symbols attribute of variable length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	sizemember		The actual number of items in the #t_symbol* array at any given moment.
        @param	maxsize			The maximum number of items in the #t_symbol* array, i.e. the number of members allocated for the array in the struct.
    */
    #define CLASS_ATTR_SYM_VARSIZE(c,attrname,flags,structname,structmember,sizemember,maxsize) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("symbol"),(maxsize),(flags),(method)0L,(method)0L,calcoffset(structname,sizemember),calcoffset(structname,structmember)))


    /**
        Create an array-of-atoms attribute of variable length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	sizemember		The actual number of items in the #t_atom array at any given moment.
        @param	maxsize			The maximum number of items in the #t_atom array, i.e. the number of members allocated for the array in the struct.
    */
    #define CLASS_ATTR_ATOM_VARSIZE(c,attrname,flags,structname,structmember,sizemember,maxsize) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("atom"),(maxsize),(flags),(method)0L,(method)0L,calcoffset(structname,sizemember),calcoffset(structname,structmember)))


    /**
        Create an array-of-objects attribute of variable length, and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
        @param	sizemember		The actual number of items in the #t_object* array at any given moment.
        @param	maxsize			The maximum number of items in the #t_object* array, i.e. the number of members allocated for the array in the struct.
    */
    #define CLASS_ATTR_OBJ_VARSIZE(c,attrname,flags,structname,structmember,sizemember,maxsize) \
        class_addattr((c),attr_offset_array_new(attrname, c74::max::gensym("object"),(maxsize),(flags),(method)0L,(method)0L,calcoffset(structname,sizemember),calcoffset(structname,structmember)))


    /**
        Specify custom accessor methods for an attribute.
        If you specify a non-NULL value for the setter or getter,
        then the function you specify will be called to set or get the attribute's value
        rather than using the built-in accessor.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	getter			An appropriate getter method as discussed in @ref attribute_accessors,
                                or NULL to use the default getter.
        @param	setter			An appropriate setter method as discussed in @ref attribute_accessors,
                                or NULL to use the default setter.
    */
    #define CLASS_ATTR_ACCESSORS(c,attrname,getter,setter) \
        { t_object* theattr=(t_object* )class_attr_get(c,gensym(attrname)); \
            object_method(theattr,gensym("setmethod"), (void*)gensym("get"), (void*)getter); \
            object_method(theattr,gensym("setmethod"), (void*)gensym("set"), (void*)setter); }


    /**
        Add a filter to the attribute to limit the lower bound of a value.
        The limiting will be performed by the default attribute accessor.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	minval			The minimum acceptable value to which the attribute will be limited.
        @see	CLASS_ATTR_FILTER_MAX
        @see	CLASS_ATTR_FILTER_CLIP
        @see	CLASS_ATTR_MIN
    */
    #define CLASS_ATTR_FILTER_MIN(c,attrname,minval) \
        { t_object* theattr=(t_object* )class_attr_get(c,gensym(attrname)); \
            attr_addfilter_clip(theattr,minval,0,1,0); }


    /**
        Add a filter to the attribute to limit the upper bound of a value.
        The limiting will be performed by the default attribute accessor.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	maxval			The maximum acceptable value to which the attribute will be limited.
        @see	CLASS_ATTR_FILTER_MIN
        @see	CLASS_ATTR_FILTER_CLIP
        @see	CLASS_ATTR_MAX
    */
    #define CLASS_ATTR_FILTER_MAX(c,attrname,maxval) \
        { t_object* theattr=(t_object* )class_attr_get(c,gensym(attrname)); \
            attr_addfilter_clip(theattr,0,maxval,0,1); }


    /**
        Add a filter to the attribute to limit both the lower and upper bounds of a value.
        The limiting will be performed by the default attribute accessor.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	minval			The maximum acceptable value to which the attribute will be limited.
        @param	maxval			The maximum acceptable value to which the attribute will be limited.
        @see
    */
    #define CLASS_ATTR_FILTER_CLIP(c,attrname,minval,maxval) \
        { t_object* theattr=(t_object* )class_attr_get(c,gensym(attrname)); \
            attr_addfilter_clip(theattr,minval,maxval,1,1); }


    /**
        Create a new attribute that is an alias of an existing attribute.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the actual attribute as a C-string.
        @param	aliasname		The name of the new alias attribute.
    */
    #define CLASS_ATTR_ALIAS(c,attrname,aliasname) \
        {	t_object* thealias; \
            t_object* theattr=(t_object* )class_attr_get(c,gensym(attrname)); \
            thealias = object_clone(theattr); \
            object_method(thealias, c74::max::gensym("setname"),gensym(aliasname)); \
            class_addattr(c,thealias); \
            CLASS_ATTR_ATTR_PARSE(c,aliasname,"alias", c74::max::gensym("symbol"),0,attrname); }




    // macros for attribute of attributes
    #define CLASS_ATTR_ATTR_ATOMS	c74::max::class_attr_addattr_atoms
    #define CLASS_ATTR_ATTR_PARSE	c74::max::class_attr_addattr_parse
    #define CLASS_ATTR_ATTR_FORMAT	c74::max::class_attr_addattr_format


    /**
        Add a new attribute to the specified attribute to specify a default value.
        The default value will be automatically set when the object is created only if your object uses a dictionary constructor
        with the #CLASS_FLAG_NEWDICTIONARY flag.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
    */
    #define CLASS_ATTR_DEFAULT(c,attrname,flags,parsestr) \
        { auto theattr = (c74::max::t_object*)c74::max::class_attr_get(c, c74::max::gensym(attrname)); CLASS_ATTR_ATTR_PARSE(c,attrname,"default",(c74::max::t_symbol* )c74::max::object_method(theattr, c74::max::gensym("gettype")),flags,parsestr); }


    /**
        Add a new attribute to the specified attribute to indicate that the specified attribute should be saved with the patcher.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
    */
    #define CLASS_ATTR_SAVE(c,attrname,flags) \
        CLASS_ATTR_ATTR_PARSE(c,attrname,"save", c74::max::gensym("long"),flags,"1")

    /**
        Add a new attribute to the specified attribute to indicate that it is saved by the object
        (so it does not appear in italics in the inspector).

        @ingroup	attr
        @param		c			The class pointer.
        @param		attrname	The name of the attribute as a C-string.
        @param		flags		Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
     */
    #define CLASS_ATTR_SELFSAVE(c,attrname,flags) \
    CLASS_ATTR_ATTR_PARSE(c,attrname,"selfsave", c74::max::gensym("long"),flags,"1")

    /**
        A convenience wrapper for both #CLASS_ATTR_DEFAULT and #CLASS_ATTR_SAVE.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
        @see	CLASS_ATTR_DEFAULT
        @see	CLASS_ATTR_SAVE
    */
    #define CLASS_ATTR_DEFAULT_SAVE(c,attrname,flags,parsestr) \
        { CLASS_ATTR_DEFAULT(c,attrname,flags,parsestr); CLASS_ATTR_SAVE(c,attrname,flags); }


    /**
        Add a new attribute to the specified attribute to specify a default value, based on Max's Object Defaults.
        If a value is present in Max's Object Defaults, then that value will be used as the default value.
        Otherwise, use the default value specified here.
        The default value will be automatically set when the object is created only if your object uses a dictionary constructor
        with the #CLASS_FLAG_NEWDICTIONARY flag.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
    */
    #define CLASS_ATTR_DEFAULTNAME(c,attrname,flags,parsestr) \
        { auto theattr=(c74::max::t_object* )c74::max::class_attr_get(c,c74::max::gensym(attrname)); CLASS_ATTR_ATTR_PARSE(c,attrname,"defaultname",(c74::max::t_symbol* )object_method(theattr, c74::max::gensym("gettype")),flags,parsestr); }


    /**
        A convenience wrapper for both #CLASS_ATTR_DEFAULTNAME and #CLASS_ATTR_SAVE.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
        @see	CLASS_ATTR_DEFAULTNAME
        @see	CLASS_ATTR_SAVE
    */
    #define CLASS_ATTR_DEFAULTNAME_SAVE(c,attrname,flags,parsestr) \
        { CLASS_ATTR_DEFAULTNAME(c,attrname,flags,parsestr); CLASS_ATTR_SAVE(c,attrname,flags); }


    /**
        Add a new attribute to the specified attribute to specify a lower range.
        The values will not be automatically limited.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
        @see	CLASS_ATTR_MAX
        @see	CLASS_ATTR_FILTER_MAX
        @see	CLASS_ATTR_FILTER_CLIP
    */
    #define CLASS_ATTR_MIN(c,attrname,flags,parsestr) \
      { t_object* theattr=(t_object* )class_attr_get(c,gensym(attrname)); CLASS_ATTR_ATTR_PARSE(c,attrname,"min",(c74::max::t_symbol*)object_method(theattr, c74::max::gensym("gettype")),flags,parsestr); }


    /**
        Add a new attribute to the specified attribute to specify an upper range.
        The values will not be automatically limited.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
        @see	CLASS_ATTR_MIN
        @see	CLASS_ATTR_FILTER_MAX
        @see	CLASS_ATTR_FILTER_CLIP
    */
    #define CLASS_ATTR_MAX(c,attrname,flags,parsestr) \
      { t_object* theattr=(t_object* )class_attr_get(c,gensym(attrname)); CLASS_ATTR_ATTR_PARSE(c,attrname,"max",(t_symbol* )object_method(theattr, c74::max::gensym("gettype")),flags,parsestr); }


    // useful attr attr macros for UI objects

    /**
        Add a new attribute indicating that any changes to the specified attribute will trigger a call
        to the object's paint method.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
    */
    #define CLASS_ATTR_PAINT(c,attrname,flags) \
        CLASS_ATTR_ATTR_PARSE(c,attrname,"paint", c74::max::gensym("long"),flags,"1")


    /**
        A convenience wrapper for both #CLASS_ATTR_DEFAULT and #CLASS_ATTR_PAINT.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
        @see	CLASS_ATTR_DEFAULT
        @see	CLASS_ATTR_PAINT
    */
    #define CLASS_ATTR_DEFAULT_PAINT(c,attrname,flags,parsestr) \
        { CLASS_ATTR_DEFAULT(c,attrname,flags,parsestr); CLASS_ATTR_PAINT(c,attrname,flags); }


    /**
        A convenience wrapper for #CLASS_ATTR_DEFAULT, #CLASS_ATTR_SAVE, and #CLASS_ATTR_PAINT.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
        @see	CLASS_ATTR_DEFAULT
        @see	CLASS_ATTR_PAINT
        @see	CLASS_ATTR_SAVE
    */
    #define CLASS_ATTR_DEFAULT_SAVE_PAINT(c,attrname,flags,parsestr) \
        { CLASS_ATTR_DEFAULT(c,attrname,flags,parsestr); CLASS_ATTR_SAVE(c,attrname,flags); CLASS_ATTR_PAINT(c,attrname,flags); }


    /**
        A convenience wrapper for #CLASS_ATTR_DEFAULTNAME, #CLASS_ATTR_SAVE, and #CLASS_ATTR_PAINT.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
        @see	CLASS_ATTR_DEFAULTNAME
        @see	CLASS_ATTR_PAINT
        @see	CLASS_ATTR_SAVE
    */
    #define CLASS_ATTR_DEFAULTNAME_PAINT(c,attrname,flags,parsestr) \
        { CLASS_ATTR_DEFAULTNAME(c,attrname,flags,parsestr); CLASS_ATTR_PAINT(c,attrname,flags); }


    /**
        A convenience wrapper for #CLASS_ATTR_DEFAULTNAME, #CLASS_ATTR_SAVE, and #CLASS_ATTR_PAINT.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
        @see	CLASS_ATTR_DEFAULTNAME
        @see	CLASS_ATTR_PAINT
        @see	CLASS_ATTR_SAVE
    */
    #define CLASS_ATTR_DEFAULTNAME_SAVE_PAINT(c,attrname,flags,parsestr) \
        { CLASS_ATTR_DEFAULTNAME(c,attrname,flags,parsestr); CLASS_ATTR_SAVE(c,attrname,flags); CLASS_ATTR_PAINT(c,attrname,flags); }


    // useful attr attr macros for inpector properties


    /**
        Add a new attribute to the specified attribute to specify an editor style for the Max inspector.
        Available styles include
        <ul>
             <li>"text"      : a text editor</li>
            <li>"onoff"     : a toggle switch</li>
            <li>"rgba"      : a color chooser</li>
            <li>"enum"      : a menu of available choices, whose symbol will be passed upon selection</li>
            <li>"enumindex" : a menu of available choices, whose index will be passed upon selection</li>
            <li>"rect"      : a style for displaying and editing #t_rect values</li>
            <li>"font"      : a font chooser</li>
            <li>"file"      : a file chooser dialog</li>
        </ul>

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
    */
    #define CLASS_ATTR_STYLE(c,attrname,flags,parsestr) \
        CLASS_ATTR_ATTR_PARSE(c,attrname,"style", c74::max::gensym("symbol"),flags,parsestr)


    /**
        Add a new attribute to the specified attribute to specify an a human-friendly label for the Max inspector.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	labelstr		A C-string, which will be parsed into an array of atoms to set the initial value.
    */
    #define CLASS_ATTR_LABEL(c,attrname,flags,labelstr) \
        CLASS_ATTR_ATTR_FORMAT(c,attrname,"label", c74::max::gensym("symbol"),flags,"s",c74::max::gensym_tr(labelstr))


    /**
        Add a new attribute to the specified attribute to specify a list of choices to display in a menu
        for the Max inspector.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.

        @remark This macro automatically calls
        @code
        CLASS_ATTR_STYLE(c,attrname,flags,"enum").
        @endcode

        @see	CLASS_ATTR_ENUMINDEX
    */
    #define CLASS_ATTR_ENUM(c,attrname,flags,parsestr) \
        { CLASS_ATTR_STYLE(c,attrname,flags,"enum"); CLASS_ATTR_ATTR_PARSE(c,attrname,"enumvals", c74::max::gensym("atom"),flags,parsestr); }


    /**
        Add a new attribute to the specified attribute to specify a list of choices to display in a menu
        for the Max inspector.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.

        @remark This macro automatically calls
        @code
        CLASS_ATTR_STYLE(c,attrname,flags,"enumindex").
        @endcode

        @see	CLASS_ATTR_ENUM
    */
    #define CLASS_ATTR_ENUMINDEX(c,attrname,flags,parsestr) \
        { CLASS_ATTR_STYLE(c,attrname,flags,"enumindex"); CLASS_ATTR_ATTR_PARSE(c,attrname,"enumvals", c74::max::gensym("atom"),flags,parsestr); }

    // localizable versions
    #define CLASS_ATTR_ENUMINDEX2(c,attrname,flags,enum1,enum2) \
    { t_atom aaa[2]; CLASS_ATTR_STYLE(c,attrname,flags,"enumindex"); atom_setsym(aaa,c74::max::gensym_tr(enum1)); atom_setsym(aaa+1,c74::max::gensym_tr(enum2)); \
    CLASS_ATTR_ATTR_ATOMS(c,attrname,"enumvals", c74::max::gensym("atom"),flags,2,aaa); }

    #define CLASS_ATTR_ENUMINDEX3(c,attrname,flags,enum1,enum2,enum3) \
    { t_atom aaa[3]; CLASS_ATTR_STYLE(c,attrname,flags,"enumindex"); atom_setsym(aaa,c74::max::gensym_tr(enum1)); atom_setsym(aaa+1,c74::max::gensym_tr(enum2)); atom_setsym(aaa+2,c74::max::gensym_tr(enum3));\
    CLASS_ATTR_ATTR_ATOMS(c,attrname,"enumvals", c74::max::gensym("atom"),flags,3,aaa); }

    #define CLASS_ATTR_ENUMINDEX4(c,attrname,flags,enum1,enum2,enum3,enum4) \
    { t_atom aaa[4]; CLASS_ATTR_STYLE(c,attrname,flags,"enumindex"); atom_setsym(aaa,c74::max::gensym_tr(enum1)); atom_setsym(aaa+1,c74::max::gensym_tr(enum2)); atom_setsym(aaa+2,c74::max::gensym_tr(enum3)); atom_setsym(aaa+3,c74::max::gensym_tr(enum4));\
    CLASS_ATTR_ATTR_ATOMS(c,attrname,"enumvals", c74::max::gensym("atom"),flags,4,aaa); }

    #define CLASS_ATTR_ENUMINDEX5(c,attrname,flags,enum1,enum2,enum3,enum4,enum5) \
    { t_atom aaa[5]; CLASS_ATTR_STYLE(c,attrname,flags,"enumindex"); atom_setsym(aaa,c74::max::gensym_tr(enum1)); atom_setsym(aaa+1,c74::max::gensym_tr(enum2)); atom_setsym(aaa+2,c74::max::gensym_tr(enum3));\
    atom_setsym(aaa+3,c74::max::gensym_tr(enum4)); atom_setsym(aaa+4,c74::max::gensym_tr(enum5));\
    CLASS_ATTR_ATTR_ATOMS(c,attrname,"enumvals", c74::max::gensym("atom"),flags,5,aaa); }

    #define CLASS_ATTR_ENUMINDEX6(c,attrname,flags,enum1,enum2,enum3,enum4,enum5,enum6) \
    { t_atom aaa[6]; CLASS_ATTR_STYLE(c,attrname,flags,"enumindex"); atom_setsym(aaa,c74::max::gensym_tr(enum1)); atom_setsym(aaa+1,c74::max::gensym_tr(enum2)); atom_setsym(aaa+2,c74::max::gensym_tr(enum3));\
    atom_setsym(aaa+3,c74::max::gensym_tr(enum4)); atom_setsym(aaa+4,c74::max::gensym_tr(enum5)); atom_setsym(aaa+5,c74::max::gensym_tr(enum6));\
    CLASS_ATTR_ATTR_ATOMS(c,attrname,"enumvals", c74::max::gensym("atom"),flags,6,aaa); }

    #define CLASS_ATTR_ENUMINDEX7(c,attrname,flags,enum1,enum2,enum3,enum4,enum5,enum6,enum7) \
    { t_atom aaa[7]; CLASS_ATTR_STYLE(c,attrname,flags,"enumindex"); atom_setsym(aaa,c74::max::gensym_tr(enum1)); atom_setsym(aaa+1,c74::max::gensym_tr(enum2)); atom_setsym(aaa+2,c74::max::gensym_tr(enum3));\
    atom_setsym(aaa+3,c74::max::gensym_tr(enum4)); atom_setsym(aaa+4,c74::max::gensym_tr(enum5)); atom_setsym(aaa+5,c74::max::gensym_tr(enum6)); atom_setsym(aaa+6,c74::max::gensym_tr(enum7));\
    CLASS_ATTR_ATTR_ATOMS(c,attrname,"enumvals", c74::max::gensym("atom"),flags,7,aaa); }

    /**
        Add a new attribute to the specified attribute to specify a category to which the attribute is assigned
        in the Max inspector.
        Categories are represented in the inspector as tabs.
        If the specified category does not exist then it will be created.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.
    */
    #define CLASS_ATTR_CATEGORY(c,attrname,flags,parsestr) \
        CLASS_ATTR_ATTR_PARSE(c,attrname,"category", c74::max::gensym("symbol"),flags, c74::max::str_tr(parsestr))


    /**
        A convenience wrapper for #CLASS_ATTR_STYLE, and #CLASS_ATTR_LABEL.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	stylestr		A C-string that names the style for the attribute.
                                See #CLASS_ATTR_STYLE for the available styles.
        @param	labelstr		A C-string that names the category to which the attribute is assigned in the inspector.

        @see	CLASS_ATTR_STYLE
        @see	CLASS_ATTR_LABEL
    */
    #define CLASS_ATTR_STYLE_LABEL(c,attrname,flags,stylestr,labelstr) \
        { CLASS_ATTR_ATTR_PARSE(c,attrname,"style", c74::max::gensym("symbol"),flags,stylestr); CLASS_ATTR_ATTR_FORMAT(c,attrname,"label", c74::max::gensym("symbol"),flags,"s",c74::max::gensym_tr(labelstr)); }


    /**
        Add a new attribute to the specified attribute to flag an attribute as invisible to the Max inspector.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
    */
    #define CLASS_ATTR_INVISIBLE(c,attrname,flags) \
        CLASS_ATTR_ATTR_PARSE(c,attrname,"invisible", c74::max::gensym("long"),flags,"1")


    /**
        Add a new attribute to the specified attribute to specify a default order in which to list attributes.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.

        @remark	A value of zero indicates that there is no ordering.  Ordering values begin at 1.  For example:
        @code
        CLASS_ATTR_ORDER(c, "firstattr",	0, "1");
        CLASS_ATTR_ORDER(c, "secondattr",	0, "2");
        CLASS_ATTR_ORDER(c, "thirdattr",	0, "3");
        @endcode
    */
    #define CLASS_ATTR_ORDER(c,attrname,flags,parsestr) \
        CLASS_ATTR_ATTR_PARSE(c,attrname,"order", c74::max::gensym("long"),flags,parsestr)

    /**
        Add a new attribute to the specified attribute to specify that it should appear in the inspector's Basic tab.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of the attribute as a C-string.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.

     */
    #define CLASS_ATTR_BASIC(c,attrname,flags) \
    CLASS_ATTR_ATTR_PARSE(c,attrname,"basic", c74::max::gensym("long"),flags,"1")


    /**	Associate the name of an attribute of your class with the name of an attribute of a style.
        @ingroup	styles
        @param		c			The class whose attribute will be added to the style.
        @param		attrname	The name of the attribute of your class.
        @param		mapname		The name of the attribute from the style.
        @see		'jslider' example project in the SDK.
    */
    void class_attr_stylemap(t_class *c, const char *attrname, const char *mapname);


    /**	Enable attributes whose name matches a style to use the style,
        even if there is a custom setter/getter for the attribute.
        @ingroup	styles
        @param		c			The class whose attribute will be added to the style.
        @param		name	The name of the attribute of your class.
    */
    void class_attr_setstyle(t_class *c, const char *name);


    // useful attr attr macro for objects that embed binary data as base64

    #define CLASS_ATTR_ATOMARRAY(c,attrname,flags) \
        CLASS_ATTR_ATTR_PARSE(c,attrname,"atomarray", c74::max::gensym("long"),flags,"1")


    /**	Define and add attributes to class methods.
        @ingroup attr
        @param	c				The class pointer.
        @param	methodname		The name of the existing method as a C-string.
        @param	attrname		The name of the attribute to add as a C-string.
        @param	type			The datatype of the attribute to be added.
        @param	flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param	parsestring		A C-string, which will be parsed into an array of atoms to set the initial value.

        @remark An example which makes a method invisible to users:
        @code
        class_addmethod(c, (method)my_foo, "foo", 0);
        CLASS_METHOD_ATTR_PARSE(c, "foo", "undocumented", c74::max::gensym("long"), 0, "1");
        @endcode
    */
    #define CLASS_METHOD_ATTR_PARSE(c,methodname,attrname,type,flags,parsestring) \
        {	t_hashtab* methods=NULL; \
            t_object* m=NULL; \
            methods = (t_hashtab* )class_extra_lookup(c,gensym("methods")); \
            if (methods) { \
                hashtab_lookup(methods,gensym((methodname)),&m); \
                if (m) \
                    object_addattr_parse(m,attrname,type,flags,parsestring); \
            } \
        }


    // sticky macros for attribute attributes, and method attributes. Useful for defining attribute groups

    /**
        Create an attribute, and add it to all following attribute declarations.
        The block is closed by a call to #CLASS_STICKY_ATTR_CLEAR.

        @ingroup	attr
        @param		c				The class pointer.
        @param		name			The name of the new attribute to create as a C-string.
        @param		flags			Any flags you wish to declare for this new attribute, as defined in #e_max_attrflags.
        @param		parsestr		A C-string, which will be parsed into an array of atoms to set the initial value.

        @remark		The most common use of CLASS_STICKY_ATTR is for creating multiple attributes with the same category,
                    as in this example:
        @code
        CLASS_STICKY_ATTR(c, "category", 0, "Foo");

        CLASS_ATTR_DOUBLE(c, "bar", 0, t_myobject, x_bar);
        CLASS_ATTR_LABEL(c, "bar", 0, "A Bar");

        CLASS_ATTR_CHAR(c, "switch", 0, t_myobject, x_switch);
        CLASS_ATTR_STYLE_LABEL(c, "switch", 0, "onoff", "Bar Switch");

        CLASS_ATTR_DOUBLE(c, "flow", 0, t_myobject, x_flow);
        CLASS_ATTR_LABEL(c, "flow",	0, "Flow Amount");

        CLASS_STICKY_ATTR_CLEAR(c, "category");
        @endcode

        @see		CLASS_STICKY_ATTR_CLEAR
    */
    #define CLASS_STICKY_ATTR(c,name,flags,parsestr) \
        { t_object* attr = attribute_new_parse(name,NULL,flags,parsestr); class_sticky(c,gensym("sticky_attr"),gensym(name),attr); }


    /**
        Close a #CLASS_STICKY_ATTR block.

        @ingroup	attr
        @param		c				The class pointer.
        @param		name			The name of the sticky attribute as a C-string.
        @see		CLASS_STICKY_ATTR
    */
    #define CLASS_STICKY_ATTR_CLEAR(c,name) class_sticky_clear(c,gensym("sticky_attr"),name?gensym(name):NULL)

    #define CLASS_STICKY_CATEGORY(c,flags,name) \
    { t_object* attr = attribute_new_format("category",NULL,flags,"s",c74::max::gensym_tr(name)); class_sticky(c,gensym("sticky_attr"),gensym("category"),attr); }

    #define CLASS_STICKY_CATEGORY_CLEAR(c) class_sticky_clear(c,gensym("sticky_attr"),gensym("category"))




    //object_method_typed utilities

    /**
        Convenience wrapper for object_method_typed() that uses atom_setparse() to define the arguments.

        @ingroup	obj
        @param		x			The object to which the message will be sent.
        @param		s			The name of the method to call on the object.
        @param 		parsestr	A C-string to parse into an array of atoms to pass to the method.
        @param		rv			The address of an atom to hold a return value.
        @return		A Max error code.

        @see		object_method_typed()
        @see		atom_setparse()
    */
    t_max_err object_method_parse(t_object* x, t_symbol* s, const char* parsestr, t_atom* rv);
    t_max_err object_method_binbuf(t_object* x, t_symbol* s, void* buf, t_atom* rv);
    t_max_err object_method_attrval(t_object* x, t_symbol* s, t_symbol* attrname, t_object* obj, t_atom* rv);
    t_max_err object_method_objval(t_object* x, t_symbol* s, t_object* obj, t_atom* rv);

    /**
        Convenience wrapper for object_method_typed() that uses atom_setformat() to define the arguments.

        @ingroup	obj
        @param		x			The object to which the message will be sent.
        @param		s			The name of the method to call on the object.
        @param		rv			The address of an atom to hold a return value.
        @param		fmt			An sprintf-style format string specifying values for the atoms.
        @param		...			One or more arguments which are to be substituted into the format string.
        @return		A Max error code.

        @see		object_method_typed()
        @see		atom_setformat()
    */
    t_max_err object_method_format(t_object* x, t_symbol* s, t_atom* rv, const char* fmt, ...);



    /**
        Convenience wrapper for object_method_typed() that passes a single char as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		v		An argument to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_char(t_object* x, t_symbol* s, unsigned char v, t_atom* rv);


    /**
        Convenience wrapper for object_method_typed() that passes a single long integer as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		v		An argument to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_long(t_object* x, t_symbol* s, long v, t_atom* rv);


    /**
        Convenience wrapper for object_method_typed() that passes a single 32bit float as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		v		An argument to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_float(t_object* x, t_symbol* s, float v, t_atom* rv);


    /**
        Convenience wrapper for object_method_typed() that passes a single 64bit float as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		v		An argument to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_double(t_object* x, t_symbol* s, double v, t_atom* rv);


    /**
        Convenience wrapper for object_method_typed() that passes a single #t_symbol* as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		v		An argument to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_sym(t_object* x, t_symbol* s, t_symbol* v, t_atom* rv);


    /**
        Convenience wrapper for object_method_typed() that passes a single #t_object* as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		v		An argument to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_obj(t_object* x, t_symbol* s, t_object* v, t_atom* rv);



    /**
        Convenience wrapper for object_method_typed() that passes an array of char values as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		ac		The number of arguments to pass to the method.
        @param		av		The address of the first of the array of arguments to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_char_array(t_object* x, t_symbol* s, long ac, unsigned char* av, t_atom* rv);


    /**
        Convenience wrapper for object_method_typed() that passes an array of long integers values as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		ac		The number of arguments to pass to the method.
        @param		av		The address of the first of the array of arguments to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_long_array(t_object* x, t_symbol* s, long ac, t_atom_long* av, t_atom* rv);


    /**
        Convenience wrapper for object_method_typed() that passes an array of 32bit floats values as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		ac		The number of arguments to pass to the method.
        @param		av		The address of the first of the array of arguments to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_float_array(t_object* x, t_symbol* s, long ac, float* av, t_atom* rv);


    /**
        Convenience wrapper for object_method_typed() that passes an array of 64bit float values as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		ac		The number of arguments to pass to the method.
        @param		av		The address of the first of the array of arguments to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_double_array(t_object* x, t_symbol* s, long ac, double* av, t_atom* rv);


    /**
        Convenience wrapper for object_method_typed() that passes an array of #t_symbol* values as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		ac		The number of arguments to pass to the method.
        @param		av		The address of the first of the array of arguments to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_sym_array(t_object* x, t_symbol* s, long ac, t_symbol** av, t_atom* rv);


    /**
        Convenience wrapper for object_method_typed() that passes an array of #t_object* values as an argument.

        @ingroup	obj
        @param		x		The object to which the message will be sent.
        @param		s		The name of the method to call on the object.
        @param		ac		The number of arguments to pass to the method.
        @param		av		The address of the first of the array of arguments to pass to the method.
        @param		rv		The address of an atom to hold a return value.

        @return		A Max error code.
        @see		object_method_typed()
    */
    t_max_err object_method_obj_array(t_object* x, t_symbol* s, long ac, t_object** av, t_atom* rv);


    // call_method_typed utilities -- not currently used in any Cycling '74 code

    t_max_err call_method_typed(method m, t_object* x, t_symbol* s, long ac, t_atom* av, t_atom* rv);
    t_max_err call_method_parse(method m, t_object* x, t_symbol* s, char* parsestr, t_atom* rv);
    t_max_err call_method_binbuf(method m, t_object* x, t_symbol* s, void* buf, t_atom* rv);
    t_max_err call_method_attrval(method m, t_object* x, t_symbol* s, t_symbol* attrname, t_object* obj, t_atom* rv);
    t_max_err call_method_objval(method m, t_object* x, t_symbol* s, t_object* obj, t_atom* rv);
    t_max_err call_method_format(method m, t_object* x, t_symbol* s, t_atom* rv, char* fmt, ...);

    t_max_err call_method_char(method m, t_object* x, t_symbol* s, unsigned char v, t_atom* rv);
    t_max_err call_method_long(method m, t_object* x, t_symbol* s, long v, t_atom* rv);
    t_max_err call_method_float(method m, t_object* x, t_symbol* s,float v, t_atom* rv);
    t_max_err call_method_double(method m, t_object* x, t_symbol* s, double v, t_atom* rv);
    t_max_err call_method_sym(method m, t_object* x, t_symbol* s, t_symbol* v, t_atom* rv);
    t_max_err call_method_obj(method m, t_object* x, t_symbol* s, t_object* v, t_atom* rv);

    t_max_err call_method_char_array(method m, t_object* x, t_symbol* s, long ac, unsigned char* av, t_atom* rv);
    t_max_err call_method_long_array(method m, t_object* x, t_symbol* s, long ac, t_atom_long* av, t_atom* rv);
    t_max_err call_method_float_array(method m, t_object* x, t_symbol* s, long ac, float* av, t_atom* rv);
    t_max_err call_method_double_array(method m, t_object* x, t_symbol* s, long ac, double* av, t_atom* rv);
    t_max_err call_method_sym_array(method m, t_object* x, t_symbol* s, long ac, t_symbol** av, t_atom* rv);
    t_max_err call_method_obj_array(method m, t_object* x, t_symbol* s, long ac, t_object** av, t_atom* rv);


    // object attribute methods (will move to attribtue_util.c with the rest of these)

    /**
        Set an attribute value with one or more atoms parsed from a C-string.

        @ingroup	attr
        @param		x			The object whose attribute will be set.
        @param		s			The name of the attribute to set.
        @param 		parsestr	A C-string to parse into an array of atoms to set the attribute value.
        @return		A Max error code.
        @see		atom_setparse()
    */
    t_max_err object_attr_setparse(t_object* x, t_symbol* s, const char* parsestr);
    t_max_err object_attr_setbinbuf(t_object* x, t_symbol* s, void* buf);
    t_max_err object_attr_setattrval(t_object* x, t_symbol* s, t_symbol* attrname, t_object* obj);
    t_max_err object_attr_setobjval(t_object* x, t_symbol* s, t_object* obj);
    t_max_err object_attr_setformat(t_object* x, t_symbol* s, const char* fmt, ...);


    // t_attribute these probably belong in attribute.c
    t_object* attribute_new_atoms(const char* attrname, t_symbol* type, long flags, long ac, t_atom* av);
    t_object* attribute_new_parse(const char* attrname, t_symbol* type, long flags, const char* parsestr);
    t_object* attribute_new_binbuf(const char* attrname, t_symbol* type, long flags, void* buf);
    t_object* attribute_new_attrval(const char* attrname, t_symbol* type, long flags, t_symbol* objattrname, t_object* obj);
    t_object* attribute_new_objval(const char* attrname, t_symbol* type, long flags, t_object* obj);
    t_object* attribute_new_format(const char* attrname, t_symbol* type, long flags, const char* fmt, ...);


    // general object constructors for objects with typed constructors

    /**
        Create a new object with one or more atoms parsed from a C-string.
        The object's new method must have an #A_GIMME signature.

        @ingroup	attr
        @param		name_space	The namespace in which to create the instance. Typically this is either #CLASS_BOX or #CLASS_NOBOX.
        @param		classname	The name of the class to instantiate.
        @param 		parsestr	A C-string to parse into an array of atoms to set the attribute value.

        @return		A pointer to the new instance.
        @see		atom_setparse()
        @see		object_new_typed()
    */
    void* object_new_parse(t_symbol* name_space, t_symbol* classname, const char* parsestr);
    void* object_new_binbuf(t_symbol* name_space, t_symbol* classname, void* buf);
    void* object_new_attrval(t_symbol* name_space, t_symbol* classname, t_symbol* objattrname, t_object* obj);
    void* object_new_objval(t_symbol* name_space, t_symbol* classname, t_object* obj);
    void* object_new_format(t_symbol* name_space, t_symbol* classname, const char* fmt, ...);	// not used or tested in any Cycling '74 code


    // attr attr functions necessary due to offset attributes as singletons for the class
    // need to copy object local to set any attributes.
    // undocumented for now in favor of using the macros defined above.

    t_max_err object_attr_addattr(t_object* x, t_symbol* attrname, t_object* attr);
    t_object* object_attr_attr_get(t_object* x, t_symbol* attrname, t_symbol* attrname2);
    t_max_err object_attr_attr_setvalueof(t_object* x, t_symbol* attrname, t_symbol* attrname2, long argc, t_atom* argv);
    t_max_err object_attr_attr_getvalueof(t_object* x, t_symbol* attrname, t_symbol* attrname2, long* argc, t_atom** argv);

    t_max_err class_attr_addattr(t_class* c, t_symbol* attrname, t_object* attr);
    t_object* class_attr_attr_get(t_class* c, t_symbol* attrname, t_symbol* attrname2);
    t_max_err class_attr_attr_setvalueof(t_class* c, t_symbol* attrname, t_symbol* attrname2, long argc, t_atom* argv);
    t_max_err class_attr_attr_getvalueof(t_class* c, t_symbol* attrname, t_symbol* attrname2, long* argc, t_atom** argv);

    t_max_err object_attr_enforcelocal(t_object* x, t_symbol* attrname);

    t_max_err class_addattr_atoms(t_class* c, const char* attrname, t_symbol* type, long flags, long ac, t_atom* av);
    t_max_err class_addattr_parse(t_class* c, const char* attrname, t_symbol* type, long flags, const char* parsestr);
    t_max_err class_addattr_format(t_class* c, const char* attrname, t_symbol* type, long flags, const char* fmt, ...);
    t_max_err class_attr_addattr_atoms(t_class* c, const char* attrname, const char* attrname2, t_symbol* type, long flags, long ac, t_atom* av);
    t_max_err class_attr_addattr_parse(t_class* c, const char* attrname, const char* attrname2, t_symbol* type, long flags, const char* parsestr);
    t_max_err class_attr_addattr_format(t_class* c, const char* attrname, const char* attrname2, const t_symbol* type, long flags, const char* fmt, ...);

    t_max_err object_addattr_atoms(t_object* x, const char* attrname, t_symbol* type, long flags, long ac, t_atom* av);
    t_max_err object_addattr_parse(t_object* x, const char* attrname, t_symbol* type, long flags, const char* parsestr);
    t_max_err object_addattr_format(t_object* x, const char* attrname, t_symbol* type, long flags, const char* fmt, ...);
    t_max_err object_attr_addattr_atoms(t_object* x, const char* attrname, const char* attrname2, t_symbol* type, long flags, long ac, t_atom* av);
    t_max_err object_attr_addattr_parse(t_object* x, const char* attrname, const char* attrname2, t_symbol* type, long flags, const char* parsestr);
    t_max_err object_attr_addattr_format(t_object* x, const char* attrname, const char* attrname2, t_symbol* type, long flags, const char* fmt, ...);


    // other general functions from obex.c
    t_object* object_clone(t_object* x);
    t_object* object_clone_generic(t_object* x);





    /**	The namespace for all Max object classes which can be instantiated in a box, i.e. in a patcher.
        @ingroup class */
    static const t_symbol* CLASS_BOX = c74::max::gensym("box");

    /**	A namespace for creating hidden or internal object classes which are not a direct part of the user
        creating patcher.
        @ingroup class */
    static const t_symbol* CLASS_NOBOX = c74::max::gensym("nobox");


    /** Attribute flags
        @ingroup attr

        @remark 	To create a readonly attribute, for example,
                    you should pass ATTR_SET_OPAQUE or ATTR_SET_OPAQUE_USER as a flag when you create your attribute.
    */
    enum e_max_attrflags {
        ATTR_FLAGS_NONE =		0x0000000,	///< No flags
        ATTR_GET_OPAQUE =		0x00000001,	///< The attribute cannot be queried by either max message when used inside of a CLASS_BOX object, nor from C code.
        ATTR_SET_OPAQUE =		0x00000002, ///< The attribute cannot be set by either max message when used inside of a CLASS_BOX object, nor from C code.
        ATTR_GET_OPAQUE_USER =	0x00000100, ///< The attribute cannot be queried by max message when used inside of a CLASS_BOX object, but <em>can</em> be queried from C code.
        ATTR_SET_OPAQUE_USER =	0x00000200, ///< The attribute cannot be set by max message when used inside of a CLASS_BOX object, but <em>can</em> be set from C code.
        ATTR_GET_DEFER =		0x00010000,	// Placeholder for potential future functionality: Any attribute queries will be called through a defer().
        ATTR_GET_USURP =		0x00020000,	// Placeholder for potential future functionality: Any calls to query the attribute will be called through the equivalent of a defer(), repeated calls will be ignored until the getter is actually run.
        ATTR_GET_DEFER_LOW =	0x00040000, // Placeholder for potential future functionality: Any attribute queries will be called through a defer_low().
        ATTR_GET_USURP_LOW =	0x00080000, // Placeholder for potential future functionality: Any calls to query the attribute will be called through the equivalent of a defer_low(), repeated calls will be ignored until the getter is actually run.
        ATTR_SET_DEFER =		0x01000000, // Placeholder for potential future functionality: The attribute setter will be called through a defer().
        ATTR_SET_USURP =		0x02000000,	// Placeholder for potential future functionality: Any calls to set the attribute will be called through the equivalent of a defer_low(), repeated calls will be ignored until the setter is actually run.
        ATTR_SET_DEFER_LOW =	0x04000000, // Placeholder for potential future functionality: The attribute setter will be called through a defer_low()
        ATTR_SET_USURP_LOW =	0x08000000,	// Placeholder for potential future functionality: Any calls to set the attribute will be called through the equivalent of a defer_low(), repeated calls will be ignored until the setter is actually run.
        ATTR_IS_JBOXATTR =		0x10000000,  // a common jbox attr
        ATTR_DIRTY =			0x20000000  // attr has been changed from its default value
    };

    /** Standard values returned by function calls with a return type of #t_max_err
        @ingroup misc */
    enum e_max_errorcodes {
        MAX_ERR_NONE =			0,	///< No error
        MAX_ERR_GENERIC =		-1,	///< Generic error
        MAX_ERR_INVALID_PTR =	-2,	///< Invalid Pointer
        MAX_ERR_DUPLICATE =		-3,	///< Duplicate
        MAX_ERR_OUT_OF_MEM =	-4	///< Out of memory
    };


    /** Flags used in linklist and hashtab objects
        @ingroup datastore */
    enum e_max_datastore_flags {
        OBJ_FLAG_OBJ = 			0x00000000,	///< free using object_free()
        OBJ_FLAG_REF =			0x00000001,	///< don't free
        OBJ_FLAG_DATA =			0x00000002,	///< don't free data or call method
        OBJ_FLAG_MEMORY =		0x00000004,	///< don't call method, and when freeing use sysmem_freeptr() instead of freeobject
        OBJ_FLAG_SILENT =		0x00000100,	///< don't notify when modified
        OBJ_FLAG_INHERITABLE =	0x00000200,  ///< obexprototype entry will be inherited by subpatchers and abstractions
        OBJ_FLAG_ITERATING =	0x00001000,	///< used by linklist to signal when is inside iteration
        OBJ_FLAG_DEBUG =		0x40000000	///< context-dependent flag, used internally for linklist debug code
    };


    /**
        A method that always returns true.
        @ingroup misc
    */
    t_atom_long method_true(void* x);


    /**
        A method that always returns false.
        @ingroup misc
    */
    t_atom_long method_false(void* x);


    /**
        Initializes a class by informing Max of its name, instance creation and free functions, size and argument types.
        Developers wishing to use obex class features (attributes, etc.) <em>must</em> use class_new()
        instead of the traditional setup() function.

        @ingroup class

        @param 	name	The class's name, as a C-string
        @param 	mnew	The instance creation function
        @param 	mfree	The instance free function
        @param 	size	The size of the object's data structure in bytes.
                        Usually you use the C sizeof operator here.
        @param 	mmenu	Obsolete - pass NULL.
                         In Max 4 this was a function pointer for UI objects called when the user created a new object of the
                         class from the Patch window's palette.
        @param 	type	A standard Max <em>type list</em> as explained in Chapter 3
                         of the Writing Externals in Max document (in the Max SDK).
                         The final argument of the type list should be a 0.
                         <em>Generally, obex objects have a single type argument</em>,
                         #A_GIMME, followed by a 0.

        @return 		This function returns the class pointer for the new object class.
                        <em>This pointer is used by numerous other functions and should be
                         stored in a global or static variable.</em>
    */
    t_class* class_new(const char* name, const method mnew, const method mfree, long size, const method mmenu, short type, ...);


    /**
        Frees a previously defined object class. <em>This function is not typically used by external developers.</em>

        @ingroup class
        @param 	c		The class pointer
        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                        or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err class_free(t_class* c);


    /**
        Registers a previously defined object class. This function is required, and should be called at the end of <tt>main()</tt>.

        @ingroup class

        @param 	name_space	The desired class's name space. Typically, either the
                             constant #CLASS_BOX, for obex classes which can
                             instantiate inside of a Max patcher (e.g. boxes, UI objects,
                             etc.), or the constant #CLASS_NOBOX, for classes
                             which will only be used internally. Developers can define
                             their own name spaces as well, but this functionality is
                             currently undocumented.
        @param 	c			The class pointer

        @return 			This function returns the error code #MAX_ERR_NONE if successful,
                            or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err class_register(const t_symbol* name_space, t_class* c);


    /**
        Registers an alias for a previously defined object class.

        @ingroup class
        @param 	c			The class pointer
        @param	aliasname	A symbol who's name will become an alias for the given class

        @return 			This function returns the error code #MAX_ERR_NONE if successful,
                            or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err class_alias(t_class* c, t_symbol* aliasname);

    // function: class_copy
    /**
     * Duplicates a previously registered object class, and registers a copy of this class.
     *
     * @ingroup classmod
     *
     * @param 	src_name_space	The source class's name space.
     * @param 	src_classname	The source class's class name.
     * @param 	dst_name_space	The copied class's name space.
     * @param 	dst_classname	The copied class's class name.
     *
     * @return 	This function returns the error code <tt>MAX_ERR_NONE</tt> if successful,
     * 			or one of the other error codes defined in "ext_obex.h" if unsuccessful.
     *
     */
    t_max_err class_copy(t_symbol* src_name_space, t_symbol* src_classname, t_symbol* dst_name_space, t_symbol* dst_classname);


    /**
        Adds a method to a previously defined object class.

        @ingroup class

        @param 	c		The class pointer
        @param 	m		Function to be called when the method is invoked
        @param 	name	C-string defining the message (message selector)
        @param 	...		One or more integers specifying the arguments to the message,
                         in the standard Max type list format (see Chapter 3 of the
                         Writing Externals in Max document for more information).

        @return			This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.

        @remark 		The class_addmethod() function works essentially like the
                         traditional addmess() function, adding the function pointed to
                         by <tt>m</tt>, to respond to the message string <tt>name</tt> in the
                         leftmost inlet of the object.
    */
    t_max_err class_addmethod(t_class* c, const method m, const char* name, ...);


    /**
        Adds an attribute to a previously defined object class.

        @ingroup class

        @param 	c		The class pointer
        @param 	attr	The attribute to add. The attribute will be a pointer returned
                         by attribute_new(), attr_offset_new() or
                         attr_offset_array_new().

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err class_addattr(t_class* c, t_object* attr);


    /**
        Retrieves the name of a class, given the class's pointer.

        @ingroup class
        @param 	c		The class pointer
        @return 		If successful, this function returns the name of the class as a t_symbol* .
    */
    t_symbol* class_nameget(t_class* c);


    /**
        Finds the class pointer for a class, given the class's namespace and name.

        @ingroup class

        @param 	name_space	The desired class's name space. Typically, either the
                             constant #CLASS_BOX, for obex classes which can
                             instantiate inside of a Max patcher (e.g. boxes, UI objects,
                             etc.), or the constant #CLASS_NOBOX, for classes
                             which will only be used internally.
        @param 	classname	The name of the class to be looked up

        @return 			If successful, this function returns the class's data pointer. Otherwise, it returns NULL.
    */
    t_class* class_findbyname(t_symbol* name_space, t_symbol* classname);


    /**
        Finds the class pointer for a class, given the class's namespace and name.

        @ingroup class

        @param 	name_space	The desired class's name space. Typically, either the
                             constant #CLASS_BOX, for obex classes which can
                             instantiate inside of a Max patcher (e.g. boxes, UI objects,
                             etc.), or the constant #CLASS_NOBOX, for classes
                             which will only be used internally. Developers can define
                             their own name spaces as well, but this functionality is
                             currently undocumented.
        @param 	classname	The name of the class to be looked up (case free)

        @return 			If successful, this function returns the class's data pointer. Otherwise, it returns NULL.
    */
    t_class* class_findbyname_casefree(t_symbol* name_space, t_symbol* classname);


    /**
        Wraps user gettable attributes with a method that gets the values and sends out dumpout outlet.

        @ingroup class
        @param 	c		The class pointer
        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                        or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err class_dumpout_wrap(t_class* c);

    t_class* class_getifloaded(t_symbol* name_space, t_symbol* classname);
    t_class* class_getifloaded_casefree(const t_symbol* name_space, const t_symbol* classname);


    /**
        Determines if a particular object is an instance of a given class.

        @ingroup obj

        @param 	x		The object to test
        @param 	name	The name of the class to test this object against
        @return 		This function returns 1 if the object is an instance of the named class. Otherwise, 0 is returned.
        @remark 		For instance, to determine whether an unknown object pointer is a pointer to a print object, one would call:

        @code
        long isprint = object_classname_compare(x, c74::max::gensym("print"));
        @endcode
    */
    long object_classname_compare(void* x, t_symbol* name);

    t_hashtab* reg_object_namespace_lookup(t_symbol* name_space);
    method class_method(t_class* x, t_symbol* methodname);
    t_messlist *class_mess(t_class* x, t_symbol* methodname);
    t_messlist *object_mess(t_object* x, t_symbol* methodname);
    method class_attr_method(t_class* x, t_symbol* methodname, void** attr, long* get);
    void* class_attr_get(t_class* x, t_symbol* attrname);
    t_max_err class_extra_store(t_class* x,t_symbol* s,t_object* o);
    t_max_err class_extra_storeflags(t_class* x, t_symbol* s, t_object* o, long flags);
    void* class_extra_lookup(t_class* x, t_symbol* s);
    t_max_err class_addtypedwrapper(t_class* x, method m, const char* name, ...);
    t_messlist *class_typedwrapper_get(t_class* x, t_symbol* s);
    t_max_err object_addtypedwrapper(t_object* x, method m, char* name, ...);
    t_messlist *object_typedwrapper_get(t_object* x, t_symbol* s);
    t_hashtab* class_namespace_fromsym(t_symbol* name_space);
    t_max_err class_namespace_getclassnames(t_symbol* name_space, long* kc, t_symbol*** kv);
    t_max_err class_setpath(t_class* x, short vol);
    short class_getpath(t_class* x);


    /**
        Allocates the memory for an instance of an object class and initialize its object header.
        It is used like the traditional function newobject, inside of an object's <tt>new</tt> method, but its use is required with obex-class objects.

        @ingroup obj
        @param 	c		The class pointer, returned by class_new()
        @return 		This function returns a new instance of an object class if successful, or NULL if unsuccessful.
    */
    void* object_alloc(t_class* c);





    t_object* object_new_imp(void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, void* p7, void* p8, void* p9, void* p10);





    /**
        Allocates the memory for an instance of an object class and initialize its object header <em>internal to Max</em>.
        It is used similarly to the traditional function newinstance(), but its use is required with obex-class objects.
        The object_new_typed() function differs from object_new() by its use of an atom list for object arguments—in this way,
        it more resembles the effect of typing something into an object box from the Max interface.

        @ingroup obj

        @param 	name_space	The desired object's name space. Typically, either the
                             constant #CLASS_BOX, for obex classes which can
                             instantiate inside of a Max patcher (e.g. boxes, UI objects,
                             etc.), or the constant #CLASS_NOBOX, for classes
                             which will only be used internally. Developers can define
                             their own name spaces as well, but this functionality is
                             currently undocumented.
        @param 	classname	The name of the class of the object to be created
        @param 	ac			Count of arguments in <tt>av</tt>
        @param 	av			Array of t_atoms; arguments to the class's instance creation function.

        @return 			This function returns a new instance of the object class if successful, or NULL if unsuccessful.
    */
    void* object_new_typed(t_symbol* name_space, t_symbol* classname, long ac, t_atom* av);


    /**	Call the free function and release the memory for an instance of an internal object class previously instantiated using object_new(),
        object_new_typed() or other new-style object constructor functions (e.g. hashtab_new()).
        It is, at the time of this writing, a wrapper for the traditional function freeobject(), but its use is suggested with obex-class objects.

        @ingroup obj
        @param 	x		The pointer to the object to be freed.
        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_free(void* x);


    /**
        do a strongly typed direct call to a method of an object

        @ingroup obj


        @param  rt		The type of the return value (double, void*, void...)
        @param	sig		the actual signature of the function in brackets !
                        something like (t_object* , double, long)
        @param 	x		The object where the method we want to call will be looked for,
                        it will also always be the first argument to the function call
        @param 	s		The message selector
        @param 	...		Any arguments to the call, the first one will always be the object (x)

        @return 		will return anything that the called function returns, typed by (rt)

        @remark 		Example: To call the function identified by <tt>getcolorat</tt> on the object <tt>pwindow</tt>
                        which is declared like:
                        t_jrgba pwindow_getcolorat(t_object* window, double x, double y)
        @code
        double x = 44.73;
        double y = 79.21;
        t_object* pwindow;
        t_jrgba result = object_method_direct(t_jrgba, (t_object* , double, double), pwindow, c74::max::gensym("getcolorat"), x, y);
        @endcode
    */

    #define object_method_direct(rt, sig, x, s, ...) ((rt (*)sig)object_method_direct_getmethod((c74::max::t_object*)x, s))(c74::max::object_method_direct_getobject((c74::max::t_object*)x, s), __VA_ARGS__)

    method object_method_direct_getmethod(t_object* x, t_symbol* sym);
    t_object* object_method_direct_getobject(t_object* x, t_symbol* sym);

    /**	Sends a type-checked message to an object.

        @ingroup obj

        @param 	x		The object that will receive the message
        @param 	s		The message selector
        @param 	ac		Count of message arguments in <tt>av</tt>
        @param 	av		Array of t_atoms; the message arguments
        @param 	rv		Return value of function, if available

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.

        @remark 		If the receiver object can respond to the message, object_method_typed() returns the result in <tt>rv</tt>. Otherwise, <tt>rv</tt> will contain an #A_NOTHING atom.
    */
    t_max_err object_method_typed(void* x, t_symbol* s, long ac, t_atom* av, t_atom* rv);


    /**	Retrieves an object's #method for a particular message selector.

        @ingroup obj
        @param 	x		The object whose method is being queried
        @param 	s		The message selector
        @return 		This function returns the #method if successful, or method_false() if unsuccessful.
    */
    method object_getmethod(void* x, t_symbol* s);



    t_symbol* class_namespace(t_class* c);		// return the namespace the class is part of


    /**
        Registers an object in a namespace.

        @ingroup obj

        @param 	name_space	The namespace in which to register the object. The namespace can be any symbol.
                             If the namespace does not already exist, it is created automatically.
        @param 	s			The name of the object in the namespace. This name will be
                             used by other objects to attach and detach from the registered object.
        @param 	x			The object to register

        @return 	The function returns a pointer to the registered object. Under some
                    circumstances, object_register will <em>duplicate</em> the object,
                     and return a pointer to the duplicate—the developer should not assume
                     that the pointer passed in is the same pointer that has been registered.
                     To be safe, the returned pointer should be stored and used with the
                     bject_unregister() function.

        @remark		You should not register an object if the object is a UI object.
                    UI objects automatically register and attach to themselves in jbox_new().
    */
    void* object_register(t_symbol* name_space, t_symbol* s, void* x);


    t_symbol* object_register_unique(t_symbol* name_space, t_symbol* s, void* x);


    /**
        Determines a registered object's pointer, given its namespace and name.

        @ingroup obj

        @param 	name_space	The namespace of the registered object
        @param 	s			The name of the registered object in the namespace

        @return 	This function returns the pointer of the registered object,
                     if successful, or NULL, if unsuccessful.
    */
    void* object_findregistered(t_symbol* name_space, t_symbol* s);


    /**
        Determines the namespace and/or name of a registered object, given the object's pointer.

        @ingroup obj

        @param 	name_space	Pointer to a t_symbol* , to receive the namespace of the registered object
        @param 	s			Pointer to a t_symbol* , to receive the name of the registered object within the namespace
        @param 	x			Pointer to the registered object

        @return 	This function returns the error code #MAX_ERR_NONE if successful,
                     or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_findregisteredbyptr(t_symbol** name_space, t_symbol** s, void* x);

    /**
        Returns all registered names in a namespace

        @ingroup obj

        @param 	name_space	Pointer to a t_symbol, the namespace to lookup names in
        @param 	namecount	Pointer to a long, to receive the count of the registered names within the namespace
        @param 	names		Pointer to a t_symbol** , to receive the allocated names. This pointer should be freed after use

        @return 	This function returns the error code <tt>MAX_ERR_NONE</tt> if successful,
                    or one of the other error codes defined in "ext_obex.h" if unsuccessful.
    */
    t_max_err object_register_getnames(t_symbol* name_space, long* namecount, t_symbol*** names);

    /**
        Attaches a client to a registered object.
        Once attached, the object will receive notifications sent from the registered object (via the object_notify() function),
        if it has a <tt>notify</tt> method defined and implemented.

        @ingroup obj

        @param 	name_space	The namespace of the registered object.
                            This should be the same value used in object_register() to register the object.
                            If you don't know the registered object's namespace, the object_findregisteredbyptr() function can be used to determine it.
        @param 	s			The name of the registered object in the namespace.
                            If you don't know the name of the registered object, the object_findregisteredbyptr() function can be used to determine it.
        @param 	x			The client object to attach. Generally, this is the pointer to your Max object.

        @return 	This function returns a pointer to the registered object (to the object
                     referred to by the combination of <tt>name_space</tt> and <tt>s</tt>
                     arguments) if successful, or NULL if unsuccessful.

        @remark		You should not attach an object to itself if the object is a UI object.
                    UI objects automatically register and attach to themselves in jbox_new().

        @see		object_notify()
        @see		object_detach()
        @see		object_attach_byptr()
        @see		object_register()
    */
    void* object_attach(t_symbol* name_space, t_symbol* s, void* x);


    /**
        Detach a client from a registered object.

        @ingroup obj

        @param 	name_space	The namespace of the registered object.
                            This should be the same value used in object_register() to register the object.
                            If you don't know the registered object's namespace, the object_findregisteredbyptr() function can be used to determine it.
        @param 	s			The name of the registered object in the namespace.
                            If you don't know the name of the registered object, the object_findregisteredbyptr() function can be used to determine it.
        @param 	x			The client object to attach. Generally, this is the pointer to your Max object.

        @return				This function returns the error code #MAX_ERR_NONE if successful,
                            or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_detach(t_symbol* name_space, t_symbol* s, void* x);


    /**
        Attaches a client to a registered object.
        Unlike object_attach(), the client is specified by providing a pointer to that object
        rather than the registered name of that object.

        Once attached, the object will receive notifications sent from the registered object (via the object_notify() function),
        if it has a <tt>notify</tt> method defined and implemented.

        @ingroup obj
        @param	x					The attaching client object. Generally, this is the pointer to your Max object.
        @param	registeredobject	A pointer to the registered object to which you wish to attach.
        @return						A Max error code.

        @remark						You should not attach an object to itself if the object is a UI object.
                                    UI objects automatically register and attach to themselves in jbox_new().

        @see		object_notify()
        @see		object_detach()
        @see		object_attach()
        @see		object_register()
        @see		object_attach_byptr_register()
    */
    t_max_err object_attach_byptr(void* x, void* registeredobject);


    /**
        A convenience function wrapping object_register() and object_attach_byptr().

        @ingroup obj

        @param	x					The attaching client object. Generally, this is the pointer to your Max object.
        @param	object_to_attach	A pointer to the object to which you wish to registered and then to which to attach.
        @param	reg_name_space		The namespace in which to register the object_to_attach.
        @return						A Max error code.

        @see		object_register()
        @see		object_attach_byptr()
    */
    t_max_err object_attach_byptr_register(void* x, void* object_to_attach, const t_symbol* reg_name_space);


    /**
        Detach a client from a registered object.

        @ingroup	obj
        @param		x					The attaching client object. Generally, this is the pointer to your Max object.
        @param		registeredobject	The object from which to detach.
        @return							A Max error code.

        @see		object_detach()
        @see		object_attach_byptr()
    */
    t_max_err object_detach_byptr(void* x, void* registeredobject);

    // function: object_subscribe
    /**
     * Subscribes a client to wait for an object to register. Upon registration, the object will attach. Once attached, the object will receive notifications sent from the registered object (via the <tt>object_notify</tt> function), if it has a <tt>notify</tt> method defined and implemented. See below for more information, in the reference for <tt>object_notify</tt>.
     *
     * @ingroup obj
     *
     * @param 	name_space	The namespace of the registered object. This should be the
     *						same value used in <tt>object_register</tt> to register the
     *						object. If you don't know the registered object's namespace,
     *						the <tt>object_findregisteredbyptr</tt> function can be
     *						used to determine it.
     * @param 	s			The name of the registered object in the namespace. If you
     *						don't know the name of the registered object, the
     *						<tt>object_findregisteredbyptr</tt> function can be used to
     *						determine it.
     * @param 	classname	The classname of the registered object in the namespace to
     *						use as a filter. If NULL, then it will attach to any class
     *						of object.
     * @param 	x			The client object to attach. Generally, this is the pointer to your Max object.
     *
     * @return 	This function returns a pointer to the object if registered (to the object
     *			referred to by the combination of <tt>name_space</tt> and <tt>s</tt>
     *			arguments) if successful, or NULL if the object is not yet registered.
     *
     */
    void* object_subscribe(t_symbol* name_space, t_symbol* s, t_symbol* classname, void* x);

    // function: object_unsubscribe
    /**
     * Unsubscribe a client from a registered object, detaching if the object is registered.
     *
     * @ingroup obj
     *
     * @param 	name_space	The namespace of the registered object. This should be the
     *						same value used in <tt>object_register</tt> to register the
     *						object. If you don't know the registered object's namespace,
     *						the <tt>object_findregisteredbyptr</tt> function can be
     *						used to determine it.
     * @param 	s			The name of the registered object in the namespace. If you
     *						don't know the name of the registered object, the
     *						<tt>object_findregisteredbyptr</tt> function can be used to
     *						determine it.
     * @param 	classname	The classname of the registered object in the namespace to
     *						use as a filter. Currently unused for unsubscribe.
     * @param 	x			The client object to detach. Generally, this is the pointer to your Max object.
     *
     * @return 	This function returns the error code <tt>MAX_ERR_NONE</tt> if successful,
     *			or one of the other error codes defined in "ext_obex.h" if unsuccessful.
     *
     */
    t_max_err object_unsubscribe(t_symbol* name_space, t_symbol* s, t_symbol* classname, void* x);


    /**
        Removes a registered object from a namespace.

        @ingroup obj
        @param 	x		The object to unregister. This should be the pointer returned from the object_register() function.
        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_unregister(void* x);

    /**
        Returns all registered names in a namespace

        @ingroup obj

        @param 	name_space	Pointer to a t_symbol, the namespace to lookup names in
        @param 	namecount	Pointer to a long, to receive the count of the registered names within the namespace
        @param 	names		Pointer to a t_symbol** , to receive the allocated names. This pointer should be freed after use

        @return				This function returns the error code <tt>MAX_ERR_NONE</tt> if successful,
                            or one of the other error codes defined in "ext_obex.h" if unsuccessful.
    */
    t_max_err object_register_getnames(t_symbol* name_space, long* namecount, t_symbol*** names);


    /**
        Broadcast a message (with an optional argument) from a registered object to any attached client objects.

        @ingroup obj

        @param 	x		Pointer to the registered object
        @param 	s		The message to send
        @param 	data	An optional argument which will be passed with the message.
                         Sets this argument to NULL if it will be unused.

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.

        @remark 		In order for client objects to receive notifications, they must define and implement a special method, <tt>notify</tt>, like so:
        @code
        class_addmethod(c, (method)myobject_notify, "notify", A_CANT, 0);
        @endcode

        @remark 		The <tt>notify</tt> method should be prototyped as:
        @code
        void myobject_notify(t_myobject *x, t_symbol* s, t_symbol* msg, void* sender, void* data);
        @endcode
                        where
                        <tt>x</tt> is the pointer to the receiving object,
                        <tt>s</tt> is the name of the sending (registered) object in its namespace,
                        <tt>msg</tt> is the sent message,
                        <tt>sender</tt> is the pointer to the sending object, and
                        <tt>data</tt> is an optional argument sent with the message.
                        This value corresponds to the data argument in the object_notify() method.
    */
    t_max_err object_notify(void* x, const t_symbol* s, void* data);


    /**
        Determines the class of a given object.

        @ingroup obj
        @param	x		The object to test
        @return 		This function returns the t_class*  of the object's class, if successful, or NULL, if unsuccessful.
    */
    t_class* object_class(void* x);


    /**
        Retrieves the value of an object which supports the <tt>getvalueof/setvalueof</tt> interface. See part 2 of the pattr SDK for more information on this interface.

        @ingroup obj

        @param 	x		The object whose value is of interest
        @param 	ac		Pointer to a long variable to receive the count of arguments in <tt>av</tt>. The long variable itself should be set to 0 previous to calling this function.
        @param 	av		Pointer to a t_atom* , to receive object data. The t_atom*  itself should be set to NULL previous to calling this function.

        @return			This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.

        @remark 		Calling the object_getvalueof() function allocates memory for any data it returns.
                        It is the developer's responsibility to free it, using the freebytes() function.

        @remark 		Developers wishing to design objects which will support this function being called on them must define and implement a special method, <tt>getvalueof</tt>, like so:
        @code
        class_addmethod(c, (method)myobject_getvalueof, "getvalueof", A_CANT, 0);
        @endcode

        @remark 		The <tt>getvalueof</tt> method should be prototyped as:
        @code
        t_max_err myobject_getvalueof(t_myobject *x, long* ac, t_atom** av);
        @endcode

        @remark 		And implemented, generally, as:
        @code
        t_max_err myobj_getvalueof(t_myobj *x, long* ac, t_atom** av)
        {
            if (ac && av) {
                if (*ac && *av) {
                    // memory has been passed in; use it.
                } else {
                    // allocate enough memory for your data
                    *av = (t_atom* )getbytes(sizeof(t_atom));
                }
                *ac = 1; // our data is a single floating point value
                atom_setfloat(*av, x->objvalue);
            }
            return MAX_ERR_NONE;
        }

        @remark 		By convention, and to permit the interoperability of objects using the obex API,
                        developers should allocate memory in their <tt>getvalueof</tt> methods using the getbytes() function.
        @endcode
    */
    t_max_err object_getvalueof(void* x, long* ac, t_atom** av);


    /**
        Sets the value of an object which supports the <tt>getvalueof/setvalueof</tt> interface.

        @ingroup obj

        @param 	x		The object whose value is of interest
        @param 	ac		The count of arguments in <tt>av</tt>
        @param 	av		Array of t_atoms; the new desired data for the object

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.

        @remark 		Developers wishing to design objects which will support this function being called on them must define and implement a special method, <tt>setvalueof</tt>, like so:
        @code
        class_addmethod(c, (method)myobject_setvalueof, "setvalueof", A_CANT, 0);
        @endcode

        @remark 		The <tt>setvalueof</tt> method should be prototyped as:
        @code
        t_max_err myobject_setvalueof(t_myobject *x, long* ac, t_atom** av);
        @endcode

        @remark 		And implemented, generally, as:
        @code
        t_max_err myobject_setvalueof(t_myobject *x, long ac, t_atom* av)
        {
            if (ac && av) {
                // simulate receipt of a float value
                myobject_float(x, atom_getfloat(av));
            }
            return MAX_ERR_NONE;
        }
        @endcode
    */
    t_max_err object_setvalueof(void* x, long ac, t_atom* av);

    /**
        Returns the pointer to an attribute, given its name.

        @ingroup attr

        @param 	x			Pointer to the object whose attribute is of interest
        @param 	attrname	The attribute's name

        @return 			This function returns a pointer to the attribute, if successful, or NULL, if unsuccessful.
    */
    void* object_attr_get(void* x, t_symbol* attrname);


    /**
        Returns the method of an attribute's <tt>get</tt> or <tt>set</tt> function, as well as a pointer to the attribute itself, from a message name.

        @ingroup attr

        @param 	x			Pointer to the object whose attribute is of interest
        @param 	methodname	The Max message used to call the attribute's <tt>get</tt> or <tt>set</tt> function. For example, <tt>gensym("mode")</tt> or <tt>gensym("getthresh")</tt>.
        @param 	attr		A pointer to a void* , which will be set to the attribute pointer upon successful completion of the function
        @param 	get			A pointer to a long variable, which will be set to 1 upon successful completion of the function,
                            if the queried method corresponds to the <tt>get</tt> function of the attribute.

        @return 			This function returns the requested method, if successful, or NULL, if unsuccessful.
    */
    method object_attr_method(void* x, t_symbol* methodname, void** attr, long* get);


    /**
        Determines if an object's attribute can be set from the Max interface (i.e. if its #ATTR_SET_OPAQUE_USER flag is set).

        @ingroup attr

        @param 	x		Pointer to the object whose attribute is of interest
        @param 	s		The attribute's name

        @return 		This function returns 1 if the attribute can be set from the Max interface. Otherwise, it returns 0.
    */
    long object_attr_usercanset(void* x,t_symbol* s);


    /**
        Determines if the value of an object's attribute can be queried from the Max interface (i.e. if its #ATTR_GET_OPAQUE_USER flag is set).

        @ingroup attr

        @param 	x		Pointer to the object whose attribute is of interest
        @param 	s		The attribute's name

        @return 		This function returns 1 if the value of the attribute can be queried from the Max interface. Otherwise, it returns 0.
    */
    long object_attr_usercanget(void* x,t_symbol* s);


    /**
        Forces a specified object's attribute to send its value from the object's dumpout outlet in the Max interface.

        @ingroup attr

        @param 	x		Pointer to the object whose attribute is of interest
        @param 	s		The attribute's name
        @param 	argc	Unused
        @param 	argv	Unused
    */
    void object_attr_getdump(void* x, t_symbol* s, long argc, t_atom* argv);


    t_max_err object_attr_getvalueof(void* x, t_symbol* s, long* argc, t_atom** argv);


    /**
        Sets the value of an object's attribute.

        @ingroup attr

        @param 	x		Pointer to the object whose attribute is of interest
        @param 	s		The attribute's name
        @param 	argc	The count of arguments in <tt>argv</tt>
        @param 	argv	Array of t_atoms; the new desired data for the attribute

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_setvalueof(void* x, t_symbol* s, long argc, const t_atom* argv);


    //object specific attributes(dynamically add/delete)

    /**
        Attaches an attribute directly to an object.

        @ingroup attr

        @param 	x		An object to which the attribute should be attached
        @param 	attr	The attribute's pointer—this should be a pointer returned from attribute_new(), attr_offset_new() or attr_offset_array_new().

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_addattr(void* x, t_object* attr);


    /**
        Detach an attribute from an object that was previously attached with object_addattr().
        The function will also free all memory associated with the attribute.
        If you only wish to detach the attribute, without freeing it, see the object_chuckattr() function.

        @ingroup attr

        @param 	x			The object to which the attribute is attached
        @param 	attrsym		The attribute's name

        @return 			This function returns the error code #MAX_ERR_NONE if successful,
                             or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_deleteattr(void* x, t_symbol* attrsym);


    /**
        Detach an attribute from an object that was previously attached with object_addattr().
        This function will <em>not</em> free the attribute (use object_free() to do this manually).

        @ingroup attr

        @param 	x			The object to which the attribute is attached
        @param 	attrsym		The attribute's name

        @return 			This function returns the error code #MAX_ERR_NONE if successful,
                             or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_chuckattr(void* x, t_symbol* attrsym);


    // obex

    /**
        Registers the byte-offset of the obex member of the class's data structure with the previously defined object class.
        Use of this function is required for obex-class objects. It must be called from <tt>main()</tt>.

        @ingroup class

        @param 	c			The class pointer
        @param 	offset		The byte-offset to the obex member of the object's data structure.
                             Conventionally, the macro #calcoffset is used to calculate the offset.
    */
    void class_obexoffset_set(t_class* c, long offset);


    /**
        Retrieves the byte-offset of the obex member of the class's data structure.

        @ingroup	class
        @param	c	The class pointer
        @return 	This function returns the byte-offset of the obex member of the class's data structure.
    */
    long class_obexoffset_get(t_class* c);


    /**
        Retrieves the value of a data stored in the obex.

        @ingroup obj

        @param 	x		The object pointer. This function should only be called on instantiated objects (i.e. in the <tt>new</tt> method or later), not directly on classes (i.e. in <tt>main()</tt>).
        @param 	key		The symbolic name for the data to be retrieved
        @param 	val		A pointer to a #t_object* , to be filled with the data retrieved from the obex.

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.

        @remark 		By default, pointers to the object's containing patcher and box objects are stored in the obex, under the keys '#P' and '#B', respectively.
                        To retrieve them, the developer could do something like the following:
        @code
        void post_containers(t_obexobj *x)
        {
            t_patcher *p;
            t_box *b;
            t_max_err err;

            err = object_obex_lookup(x, c74::max::gensym("#P"), (t_object** )&p);
            err = object_obex_lookup(x, c74::max::gensym("#B"), (t_object** )&b);

            post("my patcher is located at 0x%X", p);
            post("my box is located at 0x%X", b);
        }
        @endcode
    */
    t_max_err object_obex_lookup(void* x, t_symbol* key, t_object** val);
    t_max_err object_obex_lookuplong(void* x, t_symbol* key, t_atom_long* val);
    t_max_err object_obex_lookupsym(void* x, t_symbol* key, t_symbol** val);

    /**
        Stores data in the object's obex.

        @ingroup obj

        @param 	x		The object pointer. This function should only be called on instantiated objects (i.e. in the <tt>new</tt> method or later), not directly on classes (i.e. in <tt>main()</tt>).
        @param 	key		A symbolic name for the data to be stored
        @param 	val		A #t_object* , to be stored in the obex, referenced under the <tt>key</tt>.

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.

        @remark 		Most developers will need to use this function for the specific purpose of storing the dumpout outlet in the obex
                        (the dumpout outlet is used by attributes to report data in response to 'get' queries).
                        For this, the developer should use something like the following in the object's <tt>new</tt> method:
        @code
        object_obex_store(x, _sym_dumpout, outlet_new(x, NULL));
        @endcode
    */
    t_max_err object_obex_store(void* x,t_symbol* key, t_object* val);
    t_max_err object_obex_storeflags(void* x,t_symbol* key, t_object* val, long flags);

    t_max_err object_obex_storelong(void* x, t_symbol* key, t_atom_long val);
    t_max_err object_obex_storesym(void* x, t_symbol* key, t_symbol* val);


    /**
        Sends data from the object's dumpout outlet.
        The dumpout outlet is stored in the obex using the object_obex_store() function (see above).
        It is used approximately like outlet_anything().

        @ingroup obj

        @param 	x		The object pointer.
                        This function should only be called on instantiated objects (i.e. in the <tt>new</tt> method or later), not directly on classes (i.e. in <tt>main()</tt>).
        @param 	s		The message selector #t_symbol*
        @param 	argc	Number of elements in the argument list in argv
        @param 	argv	t_atoms constituting the message arguments

    */
    void object_obex_dumpout(void* x, const t_symbol* s, long argc, const t_atom* argv);


    // DO NOT CALL THIS -- It is called automatically now from object_free() or freeobject() -- calling this will cause problems.
    C74_DEPRECATED( void object_obex_free(void* x) );



    //attr functions

    /**
        Determines the point in an atom list where attribute arguments begin.
        Developers can use this function to assist in the manual processing of attribute arguments, when attr_args_process()
        doesn't provide the correct functionality for a particular purpose.

        @ingroup attr

        @param 	ac		The count of t_atoms in <tt>av</tt>
        @param 	av		An atom list

        @return 		This function returns an offset into the atom list, where the first attribute argument occurs.
                        For instance, the atom list <tt>foo bar 3.0 \@mode 6</tt> would cause <tt>attr_args_offset</tt> to return 3
                        (the attribute <tt>mode</tt> appears at position 3 in the atom list).
    */
    long attr_args_offset(const short ac, const t_atom* av);


    /**
        Takes an atom list and properly set any attributes described within. This function is typically used in an object's <tt>new</tt> method to conveniently process attribute arguments.

        @ingroup attr

        @param 	x		The object whose attributes will be processed
        @param 	ac		The count of t_atoms in <tt>av</tt>
        @param 	av		An atom list

        @remark 		Here is a typical example of usage:
        @code
        void* myobject_new(t_symbol* s, long ac, t_atom* av)
        {
            t_myobject *x = NULL;

            if (x=(t_myobject *)object_alloc(myobject_class))
            {
                // initialize any data before processing
                // attributes to avoid overwriting
                // attribute argument-set values
                x->data = 0;

                // process attr args, if any
                attr_args_process(x, ac, av);
            }
            return x;
        }
        @endcode
    */
    void attr_args_process(void* x, const short ac, const t_atom* av);


    //constructors


    /**
        Create a new attribute. The attribute will allocate memory and store its own data. Attributes created using attribute_new() can be assigned either to classes (using the class_addattr() function) or to objects (using the object_addattr() function).

        @ingroup attr

        @param 	name	A name for the attribute, as a C-string
        @param 	type	A t_symbol*  representing a valid attribute type.
                        At the time of this writing, the valid type-symbols are:
                        <tt>_sym_char</tt> (char),
                        <tt>_sym_long</tt> (long),
                        <tt>_sym_float32</tt> (32-bit float),
                        <tt>_sym_float64</tt> (64-bit float),
                        <tt>_sym_atom</tt> (Max #t_atom pointer),
                        <tt>_sym_symbol</tt> (Max #t_symbol pointer),
                        <tt>_sym_pointer</tt> (generic pointer) and
                        <tt>_sym_object</tt> (Max #t_object pointer).
        @param 	flags	Any attribute flags, expressed as a bitfield.
                        Attribute flags are used to determine if an attribute is accessible for setting or querying.
                        The available accessor flags are defined in #e_max_attrflags.
        @param 	mget	The method to use for the attribute's <tt>get</tt> functionality. If <tt>mget</tt> is NULL, the default method is used.
        @param 	mset	The method to use for the attribute's <tt>set</tt> functionality. If <tt>mset</tt> is NULL, the default method is used.

        @return 		This function returns the new attribute's object pointer if successful, or NULL if unsuccessful.

        @remark 		Developers wishing to define custom methods for <tt>get</tt> or <tt>set</tt> functionality need to prototype them as:
        @code
        t_max_err myobject_myattr_get(t_myobject *x, void* attr, long* ac, t_atom** av);
        @endcode
        @code
        t_max_err myobject_myattr_set(t_myobject *x, void* attr, long ac, t_atom* av);
        @endcode

        @remark 		Implementation will vary, of course, but need to follow the following basic models.
                        Note that, as with custom <tt>getvalueof</tt> and <tt>setvalueof</tt> methods for the object,
                        assumptions are made throughout Max that getbytes() has been used for memory allocation.
                        Developers are strongly urged to do the same:
        @code
        t_max_err myobject_myattr_get(t_myobject *x, void* attr, long* ac, t_atom** av)
        {
            if (*ac && *av)
                // memory passed in; use it
            else {
                *ac = 1; // size of attr data
                *av = (t_atom* )getbytes(sizeof(t_atom) * (*ac));
                if (!(*av)) {
                    *ac = 0;
                    return MAX_ERR_OUT_OF_MEM;
                }
            }
            atom_setlong(*av, x->some_value);
            return MAX_ERR_NONE;
        }

        t_max_err myobject_myattr_set(t_myobject *x, void* attr, long ac, t_atom* av)
        {
            if (ac && av) {
                x->some_value = atom_getlong(av);
            }
            return MAX_ERR_NONE;
        }
        @endcode
    */
    t_object* attribute_new(const char* name, t_symbol* type, long flags, method mget, method mset);


    /**
        Create a new attribute. The attribute references memory stored outside of itself, in the object's data structure. Attributes created using attr_offset_new() can be assigned either to classes (using the class_addattr() function) or to objects (using the object_addattr() function).

        @ingroup attr

        @param 	name	A name for the attribute, as a C-string
        @param 	type	A t_symbol*  representing a valid attribute type.
                        At the time of this writing, the valid type-symbols are:
                        <tt>_sym_char</tt> (char),
                        <tt>_sym_long</tt> (long),
                        <tt>_sym_float32</tt> (32-bit float),
                        <tt>_sym_float64</tt> (64-bit float),
                        <tt>_sym_atom</tt> (Max #t_atom pointer),
                        <tt>_sym_symbol</tt> (Max #t_symbol pointer),
                        <tt>_sym_pointer</tt> (generic pointer) and
                        <tt>_sym_object</tt> (Max #t_object pointer).
        @param 	flags	Any attribute flags, expressed as a bitfield.
                        Attribute flags are used to determine if an attribute is accessible for setting or querying.
                        The available accessor flags are defined in #e_max_attrflags.
        @param 	mget	The method to use for the attribute's <tt>get</tt> functionality.
                        If <tt>mget</tt> is NULL, the default method is used. See the discussion under attribute_new(), for more information.
        @param 	mset	The method to use for the attribute's <tt>set</tt> functionality.
                        If <tt>mset</tt> is NULL, the default method is used. See the discussion under attribute_new(), for more information.
        @param 	offset	Byte offset into the class data structure of the object which will "own" the attribute.
                        The offset should point to the data to be referenced by the attribute.
                        Typically, the #calcoffset macro (described above) is used to calculate this offset.

        @return 		This function returns the new attribute's object pointer if successful, or NULL if unsuccessful.

        @remark 		For instance, to create a new attribute which references the value of a double variable (<tt>val</tt>) in an object class's data structure:
        @code
        t_object* attr = attr_offset_new("myattr", _sym_float64 / * matches data size * /, 0 / * no flags * /, (method)0L, (method)0L, calcoffset(t_myobject, val));
        @endcode
    */
    t_object* attr_offset_new(const char* name, const t_symbol* type, long flags, const method mget, const method mset, long offset);


    /**
        Create a new attribute. The attribute references an array of memory stored outside of itself, in the object's data structure. Attributes created using attr_offset_array_new() can be assigned either to classes (using the class_addattr() function) or to objects (using the object_addattr() function).

        @ingroup attr

        @param 	name		A name for the attribute, as a C-string
        @param 	type		A t_symbol*  representing a valid attribute type.
                            At the time of this writing, the valid type-symbols are:
                            <tt>_sym_char</tt> (char),
                            <tt>_sym_long</tt> (long),
                            <tt>_sym_float32</tt> (32-bit float),
                            <tt>_sym_float64</tt> (64-bit float),
                            <tt>_sym_atom</tt> (Max #t_atom pointer),
                            <tt>_sym_symbol</tt> (Max #t_symbol pointer),
                            <tt>_sym_pointer</tt> (generic pointer) and
                            <tt>_sym_object</tt> (Max #t_object pointer).
        @param	size		Maximum number of items that may be in the array.
        @param 	flags		Any attribute flags, expressed as a bitfield.
                            Attribute flags are used to determine if an attribute is accessible for setting or querying.
                            The available accessor flags are defined in #e_max_attrflags.
        @param 	mget		The method to use for the attribute's <tt>get</tt> functionality.
                            If <tt>mget</tt> is NULL, the default method is used. See the discussion under attribute_new(), for more information.
        @param 	mset		The method to use for the attribute's <tt>set</tt> functionality.
                            If <tt>mset</tt> is NULL, the default method is used. See the discussion under attribute_new(), for more information.
        @param 	offsetcount	Byte offset into the object class's data structure of a long variable describing how many array elements
                            (up to <tt>size</tt>) comprise the data to be referenced by the attribute.
                            Typically, the #calcoffset macro is used to calculate this offset.
        @param 	offset		Byte offset into the class data structure of the object which will "own" the attribute.
                            The offset should point to the data to be referenced by the attribute.
                            Typically, the #calcoffset macro is used to calculate this offset.

        @return 			This function returns the new attribute's object pointer if successful, or NULL if unsuccessful.

        @remark 			For instance, to create a new attribute which references an array of 10 t_atoms (<tt>atm</tt>;
                            the current number of "active" elements in the array is held in the variable <tt>atmcount</tt>) in an object class's data structure:
        @code
        t_object* attr = attr_offset_array_new("myattrarray", _sym_atom / * matches data size * /, 10 / * max * /, 0 / * no flags * /, (method)0L, (method)0L, calcoffset(t_myobject, atmcount) / * count * /, calcoffset(t_myobject, atm) / * data * /);
        @endcode
    */
    t_object* attr_offset_array_new(const char* name, t_symbol* type, long size, long flags, method mget, method mset, long offsetcount, long offset);


    t_object* attr_filter_clip_new(void);


    t_object* attr_filter_proc_new(method proc);


    //for easy access of simple attributes

    /**
        Retrieves the value of an attribute, given its parent object and name.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name

        @return 		This function returns the value of the specified attribute, if successful, or 0, if unsuccessful.

        @remark 		If the attribute is not of the type specified by the function, the
                         function will attempt to coerce a valid value from the attribute.
    */
    t_atom_long object_attr_getlong(void* x, t_symbol* s);


    /**
        Sets the value of an attribute, given its parent object and name. The function will call the attribute's <tt>set</tt> method, using the data provided.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	c		An integer value; the new value for the attribute

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_setlong(void* x, t_symbol* s, t_atom_long c);


    /**
        Retrieves the value of an attribute, given its parent object and name.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name

        @return 		This function returns the value of the specified attribute, if successful, or 0, if unsuccessful.

        @remark 		If the attribute is not of the type specified by the function, the
                         function will attempt to coerce a valid value from the attribute.
    */
    t_atom_float object_attr_getfloat(void* x, t_symbol* s);


    /**
        Sets the value of an attribute, given its parent object and name. The function will call the attribute's <tt>set</tt> method, using the data provided.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	c		An floating point value; the new value for the attribute

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_setfloat(void* x, t_symbol* s, t_atom_float c);


    /**
        Retrieves the value of an attribute, given its parent object and name.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name

        @return 		This function returns the value of the specified attribute, if successful, or the empty symbol (equivalent to <tt>gensym("")</tt> or <tt>_sym_nothing</tt>), if unsuccessful.
    */
    t_symbol* object_attr_getsym(void* x, t_symbol* s);


    /**
        Sets the value of an attribute, given its parent object and name. The function will call the attribute's <tt>set</tt> method, using the data provided.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	c		A t_symbol* ; the new value for the attribute

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_setsym(void* x, t_symbol* s, t_symbol* c);


    char object_attr_getchar(void* x, t_symbol* s);
    t_max_err object_attr_setchar(void* x, t_symbol* s, char c);
    t_object* object_attr_getobj(void* x, t_symbol* s);
    t_max_err object_attr_setobj(void* x, t_symbol* s, t_object* o);


    /**
        Retrieves the value of an attribute, given its parent object and name.
        This function uses a developer-allocated array to copy data to.
        Developers wishing to retrieve the value of an attribute without pre-allocating memory should refer to the object_attr_getvalueof() function.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	max		The number of array elements in <tt>vals</tt>. The function will take care not to overwrite the bounds of the array.
        @param 	vals	Pointer to the first element of a pre-allocated array of long data.

        @return 		This function returns the number of elements copied into <tt>vals</tt>.

        @remark 		If the attribute is not of the type specified by the function, the
                         function will attempt to coerce a valid value from the attribute.
    */
    long object_attr_getlong_array(void* x, t_symbol* s, long max, t_atom_long* vals);


    /**
        Sets the value of an attribute, given its parent object and name. The function will call the attribute's <tt>set</tt> method, using the data provided.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	count	The number of array elements in vals
        @param 	vals	Pointer to the first element of an array of long data

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_setlong_array(void* x, t_symbol* s, long count, t_atom_long* vals);


    /**
        Retrieves the value of an attribute, given its parent object and name.
        This function uses a developer-allocated array to copy data to.
        Developers wishing to retrieve the value of an attribute without pre-allocating memory should refer to the object_attr_getvalueof() function.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	max		The number of array elements in <tt>vals</tt>. The function will take care not to overwrite the bounds of the array.
        @param 	vals	Pointer to the first element of a pre-allocated array of unsigned char data.

        @return 		This function returns the number of elements copied into <tt>vals</tt>.

        @remark 		If the attribute is not of the type specified by the function, the
                         function will attempt to coerce a valid value from the attribute.
    */
    long object_attr_getchar_array(void* x, t_symbol* s, long max, t_uint8 *vals);


    /**
        Sets the value of an attribute, given its parent object and name. The function will call the attribute's <tt>set</tt> method, using the data provided.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	count	The number of array elements in vals
        @param 	vals	Pointer to the first element of an array of unsigned char data

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_setchar_array(void* x, t_symbol* s, long count, const t_uint8 *vals);


    /**
        Retrieves the value of an attribute, given its parent object and name.
        This function uses a developer-allocated array to copy data to.
        Developers wishing to retrieve the value of an attribute without pre-allocating memory should refer to the object_attr_getvalueof() function.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	max		The number of array elements in <tt>vals</tt>. The function will take care not to overwrite the bounds of the array.
        @param 	vals	Pointer to the first element of a pre-allocated array of float data.

        @return 		This function returns the number of elements copied into <tt>vals</tt>.

        @remark 		If the attribute is not of the type specified by the function, the
                         function will attempt to coerce a valid value from the attribute.
    */
    long object_attr_getfloat_array(void* x, t_symbol* s, long max, float* vals);


    /**
        Sets the value of an attribute, given its parent object and name. The function will call the attribute's <tt>set</tt> method, using the data provided.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	count	The number of array elements in vals
        @param 	vals	Pointer to the first element of an array of float data

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_setfloat_array(void* x, t_symbol* s, long count, float* vals);


    /**
        Retrieves the value of an attribute, given its parent object and name.
        This function uses a developer-allocated array to copy data to.
        Developers wishing to retrieve the value of an attribute without pre-allocating memory should refer to the object_attr_getvalueof() function.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	max		The number of array elements in <tt>vals</tt>. The function will take care not to overwrite the bounds of the array.
        @param 	vals	Pointer to the first element of a pre-allocated array of double data.

        @return 		This function returns the number of elements copied into <tt>vals</tt>.

        @remark 		If the attribute is not of the type specified by the function, the
                         function will attempt to coerce a valid value from the attribute.
    */
    long object_attr_getdouble_array(void* x, t_symbol* s, long max, double* vals);


    /**
        Sets the value of an attribute, given its parent object and name. The function will call the attribute's <tt>set</tt> method, using the data provided.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	count	The number of array elements in vals
        @param 	vals	Pointer to the first element of an array of double data

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_setdouble_array(void* x, t_symbol* s, long count, double* vals);


    /**
        Retrieves the value of an attribute, given its parent object and name.
        This function uses a developer-allocated array to copy data to.
        Developers wishing to retrieve the value of an attribute without pre-allocating memory should refer to the object_attr_getvalueof() function.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	max		The number of array elements in <tt>vals</tt>. The function will take care not to overwrite the bounds of the array.
        @param 	vals	Pointer to the first element of a pre-allocated array of #t_symbol* s.

        @return 		This function returns the number of elements copied into <tt>vals</tt>.
    */
    long object_attr_getsym_array(void* x, t_symbol* s, long max, t_symbol** vals);


    /**
        Sets the value of an attribute, given its parent object and name.
        The function will call the attribute's <tt>set</tt> method, using the data provided.

        @ingroup attr

        @param 	x		The attribute's parent object
        @param 	s		The attribute's name
        @param 	count	The number of array elements in vals
        @param 	vals	Pointer to the first element of an array of #t_symbol* s

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_setsym_array(void* x, t_symbol* s, long count, t_symbol** vals);


    //attr filters util

    /**
        Attaches a clip filter to an attribute.
        The filter will <em>only</em> clip values sent to the attribute using the attribute's <tt>set</tt> function.

        @ingroup attr

        @param 	x		Pointer to the attribute to receive the filter
        @param 	min		Minimum value for the clip filter
        @param 	max		Maximum value for the clip filter
        @param 	usemin	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.
        @param 	usemax	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err attr_addfilterset_clip(void* x, double min, double max, long usemin, long usemax);


    /**
        Attaches a clip/scale filter to an attribute.
        The filter will <em>only</em> clip and scale values sent to the attribute using the attribute's <tt>set</tt> function.

        @ingroup attr

        @param 	x		Pointer to the attribute to receive the filter
        @param 	scale	Scale value. Data sent to the attribute will be scaled by this amount. <em>Scaling occurs previous to clipping</em>.
        @param 	min		Minimum value for the clip filter
        @param 	max		Maximum value for the clip filter
        @param 	usemin	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.
        @param 	usemax	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err attr_addfilterset_clip_scale(void* x, double scale, double min, double max, long usemin, long usemax);


    /**
        Attaches a clip filter to an attribute.
        The filter will <em>only</em> clip values retrieved from the attribute using the attribute's <tt>get</tt> function.

        @ingroup attr

        @param 	x		Pointer to the attribute to receive the filter
        @param 	min		Minimum value for the clip filter
        @param 	max		Maximum value for the clip filter
        @param 	usemin	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.
        @param 	usemax	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err attr_addfilterget_clip(void* x, double min, double max, long usemin, long usemax);


    /**
        Attaches a clip/scale filter to an attribute.
        The filter will <em>only</em> clip and scale values retrieved from the attribute using the attribute's <tt>get</tt> function.

        @ingroup attr

        @param 	x		Pointer to the attribute to receive the filter
        @param 	scale	Scale value. Data retrieved from the attribute will be scaled by this amount. <em>Scaling occurs previous to clipping</em>.
        @param 	min		Minimum value for the clip filter
        @param 	max		Maximum value for the clip filter
        @param 	usemin	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.
        @param 	usemax	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err attr_addfilterget_clip_scale(void* x, double scale, double min, double max, long usemin, long usemax);


    /**
        Attaches a clip filter to an attribute.
        The filter will clip any values sent to or retrieved from the attribute using the attribute's <tt>get</tt> and <tt>set</tt> functions.

        @ingroup attr

        @param 	x		Pointer to the attribute to receive the filter
        @param 	min		Minimum value for the clip filter
        @param 	max		Maximum value for the clip filter
        @param 	usemin	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.
        @param 	usemax	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err attr_addfilter_clip(void* x, double min, double max, long usemin, long usemax);


    /**
        Attaches a clip/scale filter to an attribute.
        The filter will clip and scale any values sent to or retrieved from the attribute using the attribute's <tt>get</tt> and <tt>set</tt> functions.

        @ingroup attr

        @param 	x		Pointer to the attribute to receive the filter
        @param 	scale	Scale value. Data sent to the attribute will be scaled by this amount. Data retrieved from the attribute will be scaled by its reciprocal.
                        <em>Scaling occurs previous to clipping</em>.
        @param 	min		Minimum value for the clip filter
        @param 	max		Maximum value for the clip filter
        @param 	usemin	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.
        @param 	usemax	Sets this value to 0 if the minimum clip value should <em>not</em> be used. Otherwise, set the value to non-zero.

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err attr_addfilter_clip_scale(void* x, double scale, double min, double max, long usemin, long usemax);


    /**
        Attaches a custom filter method to an attribute.
        The filter will <em>only</em> be called for values retrieved from the attribute using the attribute's <tt>set</tt> function.

        @ingroup attr

        @param 	x		Pointer to the attribute to receive the filter
        @param 	proc	A filter method

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.

        @remark 		The filter method should be prototyped and implemented as follows:
        @code
        t_max_err myfiltermethod(void* parent, void* attr, long ac, t_atom* av);

        t_max_err myfiltermethod(void* parent, void* attr, long ac, t_atom* av)
        {
            long i;
            float temp,

            // this filter rounds off all values
            // assumes that the data is float
            for (i = 0; i < ac; i++) {
                temp = atom_getfloat(av + i);
                temp = (float)((long)(temp + 0.5));
                atom_setfloat(av + i, temp);
            }
            return MAX_ERR_NONE;
        }
        @endcode
    */
    t_max_err attr_addfilterset_proc(void* x, method proc);


    /**
        Attaches a custom filter method to an attribute. The filter will <em>only</em> be called for values retrieved from the attribute using the attribute's <tt>get</tt> function.

        @ingroup attr

        @param 	x		Pointer to the attribute to receive the filter
        @param 	proc	A filter method

        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.

        @remark 		The filter method should be prototyped and implemented as described above for the attr_addfilterset_proc() function.
    */
    t_max_err attr_addfilterget_proc(void* x, method proc);


    //more util functions

    /**
        Create a dictionary of attribute-name, attribute-value pairs
        from an array of atoms containing an attribute definition list.

        @ingroup attr
        @param	x	A dictionary instance pointer.
        @param	ac	The number of atoms to parse in av.
        @param	av	A pointer to the first of the array of atoms containing the attribute values.

        @remark		The code example below shows the creation of a list of atoms using atom_setparse(),
                    and then uses that list of atoms to fill the dictionary with attr_args_dictionary().
        @code
        long ac = 0;
        t_atom* av = NULL;
        char parsebuf[4096];
        t_dictionary* d = dictionary_new();
        t_atom a;

        sprintf(parsebuf,"@defrect %.6f %.6f %.6f %.6f @title Untitled @presentation 0 ", r->x, r->y, r->width, r->height);
        atom_setparse(&ac, &av, parsebuf);
        attr_args_dictionary(d, ac, av);
        atom_setobj(&a, d);
        @endcode
    */
    void attr_args_dictionary(t_dictionary* x, short ac, t_atom* av);


    /**
        Set attributes for an object that are defined in a dictionary.
        Objects with dictionary constructors, such as UI objects,
        should call this method to set their attributes when an object is created.

        @ingroup attr
        @param	x	The object instance pointer.
        @param	d	The dictionary containing the attributes.
        @see	attr_args_process()
    */
    void attr_dictionary_process(void* x, t_dictionary* d);

    /**
        Check that a dictionary only contains values for existing attributes
        of an object. If a key in the dictionary doesn't correspond an one of
        the object's attributes, an error will be posted to the Max window.

        @ingroup attr
        @param	x	The object instance pointer.
        @param	d	The dictionary containing the attributes.
        @see	attr_dictionary_process()
    */
    void attr_dictionary_check(void* x, t_dictionary* d);


    /**
        Retrieve a pointer to a dictionary passed in as an atom argument.
        Use this function when working with classes that have dictionary constructors
        to fetch the dictionary.

        @ingroup obj
        @param	ac	The number of atoms.
        @param	av	A pointer to the first atom in the array.
        @return		The dictionary retrieved from the atoms.
        @see		attr_dictionary_process()
    */
    t_dictionary* object_dictionaryarg(const long ac, const t_atom* av);


    // use the macros for these in ext_obex_util.h
    t_max_err class_sticky(t_class* x, t_symbol* stickyname, t_symbol* s, t_object* o);
    t_max_err class_sticky_clear(t_class* x, t_symbol* stickyname, t_symbol* s);


    t_max_err object_sticky(t_object* x, t_symbol* stickyname, t_symbol* s, t_object* o);
    t_max_err object_sticky_clear(t_object* x, t_symbol* stickyname, t_symbol* s);

    t_max_err object_attr_getnames(void* x, long* argc, t_symbol*** argv);

    /**
        Allocate a single atom.
        If ac and av are both zero then memory is allocated.
        Otherwise it is presumed that memory is already allocated and nothing will happen.

        @ingroup		atom
        @param	ac		The address of a variable that will contain the number of atoms allocated (1).
        @param	av		The address of a pointer that will be set with the new allocated memory for the atom.
        @param	alloc	Address of a variable that will be set true is memory is allocated, otherwise false.
        @return			A Max error code.
    */
    t_max_err atom_alloc(long* ac, t_atom** av, char* alloc);


    /**
        Allocate an array of atoms.
        If ac and av are both zero then memory is allocated.
        Otherwise it is presumed that memory is already allocated and nothing will happen.

        @ingroup		atom
        @param	minsize	The minimum number of atoms that this array will need to contain.
                        This determines the amount of memory allocated.
        @param	ac		The address of a variable that will contain the number of atoms allocated.
        @param	av		The address of a pointer that will be set with the new allocated memory for the atoms.
        @param	alloc	Address of a variable that will be set true is memory is allocated, otherwise false.
        @return			A Max error code.
    */
    t_max_err atom_alloc_array(long minsize, long* ac, t_atom** av, char* alloc);


    /**
        Determine if a class is a user interface object.

        @ingroup 	class
        @param	c	The class pointer.
        @return		True is the class defines a user interface object, otherwise false.
    */
    long class_is_ui(t_class* c);


    /**
        Mark an attribute as being touched by some code not from the attribute setter.
        This will notify clients that the attribute has changed.

        @ingroup obj

        @param 	x			The object whose attribute has been changed
        @param 	attrname	The attribute name

        @return				A Max error code
     */
    t_max_err object_attr_touch(t_object* x, t_symbol* attrname);


    /**
     Get the disabled state for an attribute

     @ingroup obj

     @param 	x			The object owning the attribute
     @param 	attrname	The attribute name

     @return				1 if disabled, otherwise 0
     */
    t_ptr_int object_attr_getdisabled(t_object *x, t_symbol *attrname);

    /**
     Set the disabled state for an attribute

     @ingroup obj

     @param 	x			The object owning the attribute
     @param 	attrname	The attribute name
     @param     way         1 to disable attribute, 0 to enable it

     @return				A Max error code
     */
    t_max_err object_attr_setdisabled(t_object *x, t_symbol *attrname, long way);


    t_max_err object_retain(t_object *x);

    t_max_err class_parameter_init(t_class *c);
    t_max_err class_parameter_setinfo(t_class *c, int type, long ac, t_atom *av);
    t_max_err object_parameter_init_flags(t_object *x, int type, unsigned int flags);
    // set the value of a particular parameter datum
    t_max_err object_parameter_setinfo(t_object *x, int type, long ac, t_atom *av);

    t_max_err class_parameter_register_default_color(t_class* c, t_symbol* attrname, t_symbol* colorname);


    END_USING_C_LINKAGE


    /**	Send a message to an object with no arguments.
        In previous versions of the SDK this object might send messages with arguments as well.
        For those cases you should now use object_method_typed() or object_method_direct().

        @ingroup obj
     */
    inline void* object_method(t_object* target_object, t_symbol* method_name) {
        method m = object_getmethod(target_object, method_name);
        if (m)
            return m(target_object);
        else
            return nullptr;
    }

    inline void* object_method(t_object* target_object, t_symbol* method_name, void* arg1) {
        method m = object_getmethod(target_object, method_name);
        if (m)
            return m(target_object, arg1);
        else
            return nullptr;
    }

    inline void* object_method(t_object* target_object, t_symbol* method_name, void* arg1, void* arg2) {
        method m = object_getmethod(target_object, method_name);
        if (m)
            return m(target_object, arg1, arg2);
        else
            return nullptr;
    }

    inline void* object_method(t_object* target_object, t_symbol* method_name, void* arg1, void* arg2, void* arg3) {
        method m = object_getmethod(target_object, method_name);
        if (m)
            return m(target_object, arg1, arg2, arg3);
        else
            return nullptr;
    }

    inline void* object_method(t_object* target_object, t_symbol* method_name, void* arg1, void* arg2, void* arg3, void* arg4) {
        method m = object_getmethod(target_object, method_name);
        if (m)
            return m(target_object, arg1, arg2, arg3, arg4);
        else
            return nullptr;
    }

    inline void* object_method(t_object* target_object, t_symbol* method_name, void* arg1, void* arg2, void* arg3, void* arg4, void* arg5) {
        method m = object_getmethod(target_object, method_name);
        if (m)
            return m(target_object, arg1, arg2, arg3, arg4, arg5);
        else
            return nullptr;
    }


    /**
        Allocates the memory for an instance of an object class and initialize its object header <em>internal to Max</em>.
        It is used similarly to the traditional function newinstance(), but its use is required with obex-class objects.

        @ingroup obj

        @param 	name_space	The desired object's name space. Typically, either the
     constant #CLASS_BOX, for obex classes which can
     instantiate inside of a Max patcher (e.g. boxes, UI objects,
     etc.), or the constant #CLASS_NOBOX, for classes
     which will only be used internally.
        @param 	classname	The name of the class of the object to be created
        @param 	...			Any arguments expected by the object class being instantiated

        @return 			This function returns a new instance of the object class if successful, or NULL if unsuccessful.
     */
    //	t_object* object_new(const t_symbol* name_space, const t_symbol* classname, ...);
    //
    //	#ifdef C74_X64
    //		#define object_new(...) C74_VARFUN(object_new_imp, __VA_ARGS__)
    //	#endif
    //

    inline t_object* object_new(const t_symbol* name_space, const t_symbol* classname) {
        return object_new_imp((void*)name_space, (void*)classname,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    inline t_object* object_new(const t_symbol* name_space, const t_symbol* classname, void* arg1) {
        return object_new_imp((void*)name_space, (void*)classname,
                              arg1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    inline t_object* object_new(const t_symbol* name_space, const t_symbol* classname, void* arg1, void* arg2) {
        return object_new_imp((void*)name_space, (void*)classname,
                              arg1, arg2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    inline t_object* object_new(const t_symbol* name_space, const t_symbol* classname, void* arg1, void* arg2, void* arg3) {
        return object_new_imp((void*)name_space, (void*)classname,
                              arg1, arg2, arg3, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    inline t_object* object_new(const t_symbol* name_space, const t_symbol* classname, void* arg1, void* arg2, void* arg3, void* arg4) {
        return object_new_imp((void*)name_space, (void*)classname,
                              arg1, arg2, arg3, arg4, nullptr, nullptr, nullptr, nullptr);
    }

    inline t_object* object_new(const t_symbol* name_space, const t_symbol* classname, void* arg1, void* arg2, void* arg3, void* arg4, void* arg5) {
        return object_new_imp((void*)name_space, (void*)classname,
                              arg1, arg2, arg3, arg4, arg5, nullptr, nullptr, nullptr);
    }



}} // namespace c74::max
