/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {


    /** A linklist element.
        @ingroup linklist
        @see t_linklist
    */
    typedef t_object t_llelem;


    /** The linklist object.
        @ingroup linklist
        @see t_llelem
    */
    typedef t_object t_linklist;


    /**
        Comparison function pointer type.

        Methods that require a comparison function pointer to be passed in use this type.
        It should return <code>true</code> or <code>false</code> depending on the outcome of the
        comparison of the two linklist items passed in as arguments.

        @ingroup datastore
        @see linklist_match()
        @see hashtab_findfirst()
        @see indexmap_sort()
    */
    typedef long (*t_cmpfn)(void* , void* );

    BEGIN_USING_C_LINKAGE

    /**
        Create a new linklist object.
        You can free the linklist by calling object_free() on the linklist's pointer,
        or by using linklist_chuck().

        @ingroup linklist
        @return  Pointer to the new linklist object.

        @see				object_free()
        @see				linklist_chuck()
    */
    t_linklist* linklist_new(void);


    /**
        Free a linklist, but don't free the items it contains.

        The linklist can contain a variety of different types of data.
        By default, the linklist assumes that all items are max objects with a valid
        #t_object header.

        You can alter the linklist's notion of what it contains by using the
        linklist_flags() method.

        When you free the linklist by calling object_free() it then tries to free all of the items it contains.
        If the linklist is storing a custom type of data, or should otherwise not free the data it contains,
        then call linklist_chuck() to free the object instead of object_free().

        @ingroup linklist
        @param	x	The linklist object to be freed.
        @see object_free
    */
    void linklist_chuck(t_linklist* x);


    /**
        Return the number of items in a linklist object.

        @ingroup linklist

        @param	x	The linklist instance.
        @return		The number of items in the linklist object.
    */
    t_atom_long linklist_getsize(t_linklist* x);


    /**
        Return the item stored in a linklist at a specified index.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	index	The index in the linklist to fetch.  Indices are zero-based.
        @return			The item from the linklist stored at index.
                        If there is no item at the index, <code>NULL</code> is returned
    */
    void* linklist_getindex(t_linklist* x, long index);


    /**
        Return an item's index, given the item itself.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	p		The item pointer to search for in the linklist.
        @return			The index of the item given in the linklist.
                        If the item is not in the linklist #MAX_ERR_GENERIC is returned.
    */
    t_atom_long linklist_objptr2index(t_linklist* x, void* p);


    /**
        Add an item to the end of the list.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	o		The item pointer to append to the linked-list.
        @return			The updated size of the linklist after appending the new item, or -1 if the append failed.
    */
    t_atom_long linklist_append(t_linklist* x, void* o);


    /**	Insert an item into the list at the specified index.
        @ingroup linklist
        @param	x		The linklist instance.
        @param	o		The item pointer to insert.
        @param	index	The index at which to insert.  Index 0 is the head of the list.
        @return			The index of the item in the linklist, or -1 if the insert failed.
    */
    t_atom_long linklist_insertindex(t_linklist* x,  void* o, long index);


    /**	Insert an item into the list, keeping the list sorted according to a specified comparison function.
        @ingroup		linklist
        @param	x		The linklist instance.
        @param	o		The item pointer to insert.
        @param	cmpfn	A comparison function by which the list should be sorted.
        @return			The index of the new item in the linklist, or -1 if the insert failed.
    */
    long linklist_insert_sorted(t_linklist* x, void* o, long cmpfn(void* , void* ));


    /**
        Insert an item into the list after another specified item.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	o		The item pointer to insert.
        @param	objptr	The item pointer after which to insert in the list.

        @return			An opaque linklist element.
    */
    t_llelem *linklist_insertafterobjptr(t_linklist* x, void* o, void* objptr);	// inserts object o after objptr


    /**
        Insert an item into the list before another specified item.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	o		The item pointer to insert.
        @param	objptr	The item pointer before which to insert in the list.

        @return			An opaque linklist element.
    */
    t_llelem *linklist_insertbeforeobjptr(t_linklist* x, void* o, void* objptr); // inserts object o before objptr


    /**
        Move an existing item in the list to a position after another specified item in the list.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	o		The item pointer to insert.
        @param	objptr	The item pointer after which to move o in the list.

        @return			An opaque linklist element.
    */
    t_llelem *linklist_moveafterobjptr(t_linklist* x, void* o, void* objptr);    // move existing object o after objptr


    /**
        Move an existing item in the list to a position before another specified item in the list.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	o		The item pointer to insert.
        @param	objptr	The item pointer before which to move o in the list.

        @return			An opaque linklist element.
    */
    t_llelem *linklist_movebeforeobjptr(t_linklist* x, void* o, void* objptr);   // move existing object o before objptr


    /**
        Remove the item from the list at the specified index and free it.

        The linklist can contain a variety of different types of data.
        By default, the linklist assumes that all items are max objects with a valid
        #t_object header.  Thus by default, it frees items by calling object_free() on them.

        You can alter the linklist's notion of what it contains by using the
        linklist_flags() method.

        If you wish to remove an item from the linklist and free it yourself, then you
        should use linklist_chuckptr().

        @ingroup linklist

        @param	x		The linklist instance.
        @param	index	The index of the item to delete.
        @return			Returns the index number of the item delted, or -1 if the operation failed.

        @see			linklist_chuckindex
        @see			linklist_chuckobject
    */
    t_atom_long linklist_deleteindex(t_linklist* x, long index);


    /**
        Remove the item from the list at the specified index.

        You are responsible for freeing any memory associated with the item that is
        removed from the linklist.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	index	The index of the item to remove.
        @return			Returns #MAX_ERR_NONE on successful removal, otherwise returns #MAX_ERR_GENERIC

        @see			linklist_deleteindex
        @see			linklist_chuckobject
    */
    long linklist_chuckindex(t_linklist* x, long index);


    /**
        Remove the specified item from the list.

        You are responsible for freeing any memory associated with the item that is
        removed from the linklist.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	o		The pointer to the item to remove.

        @see			linklist_deleteindex
        @see			linklist_chuckindex
        @see			linklist_deleteobject
    */
    long linklist_chuckobject(t_linklist* x, void* o);


    /**
        Delete the specified item from the list.

        The object is removed from the list and deleted.
        The deletion is done with respect to any flags passed to linklist_flags.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	o		The pointer to the item to delete.

        @see			linklist_deleteindex
        @see			linklist_chuckindex
        @see			linklist_chuckobject
    */
    long linklist_deleteobject(t_linklist* x, void* o);


    /**
        Remove and free all items in the list.

        Freeing items in the list is subject to the same rules as linklist_deleteindex().
        You can alter the linklist's notion of what it contains, and thus how items are freed,
        by using the linklist_flags() method.

        @ingroup	linklist
        @param		x			The linklist instance.
    */
    void linklist_clear(t_linklist* x);


    /**
        Retrieve linklist items as an array of pointers.

        @ingroup linklist

        @param	x		The linklist instance.
        @param	a		The address of the first pointer in the array to fill.
        @param	max		The number of pointers in the array.
        @return			The number of items from the list actually returned in the array.
    */
    t_atom_long linklist_makearray(t_linklist* x, void** a, long max);


    /**
        Reverse the order of items in the linked-list.

        @ingroup linklist
        @param	x	The linklist instance.
    */
    void linklist_reverse(t_linklist* x);


    /**
        Rotate items in the linked list in circular fashion.

        @ingroup linklist
        @param	x	The linklist instance.
        @param	i	The number of positions in the list to shift items.
    */
    void linklist_rotate(t_linklist* x, long i);


    /**
        Randomize the order of items in the linked-list.

        @ingroup linklist
        @param	x	The linklist instance.
    */
    void linklist_shuffle(t_linklist* x);


    /**
        Swap the position of two items in the linked-list, specified by index.

        @ingroup linklist
        @param	x	The linklist instance.
        @param	a	The index of the first item to swap.
        @param	b	The index of the second item to swap.
    */
    void linklist_swap(t_linklist* x, long a, long b);


    /**
        Search the linked list for the first item meeting defined criteria.
        The items in the list are traversed, calling a specified comparison function on each
        until the comparison function returns true.

        @ingroup linklist
        @param	x		The linklist instance.
        @param	o		The address to pointer that will be set with the matching item.
        @param	cmpfn	The function used to determine a match in the list.
        @param	cmpdata	An argument to be passed to the #t_cmpfn.
                        This will be passed as the second of the two args to the #t_cmpfn.
                        The first arg will be the linklist item at each iteration in the list.
        @return			The index of the matching item, or -1 if no match is found.

        @remark		The following shows how to manually do what linklist_chuckobject() does.
        @code
        void* obj;
        long index;

        index = linklist_findfirst(x, &obj, #linklist_match, o);
        if(index != -1)
            linklist_chuckindex(x, index);
        @endcode

        @see linklist_match
        @see t_cmpfn
        @see linklist_findall
    */
    t_atom_long linklist_findfirst(t_linklist* x, void** o, long cmpfn(void* , void* ), void* cmpdata);


    /**
        Search the linked list for all items meeting defined criteria.
        The items in the list are traversed, calling a specified comparison function on each,
        and returning the matches in another linklist.

        @ingroup linklist
        @param	x		The linklist instance.
        @param	out		The address to a #t_linklist pointer.
                        You should initialize the pointer to NULL before calling linklist_findall().
                        A new linklist will be created internally by linklist_findall() and returned here.
        @param	cmpfn	The function used to determine a match in the list.
        @param	cmpdata	An argument to be passed to the #t_cmpfn.
                        This will be passed as the second of the two args to the #t_cmpfn.
                        The first arg will be the linklist item at each iteration in the list.

        @remark		The following example assumes you have a linklist called myLinkList, and #t_cmpfn called
                      myCmpFunction, and some sort of data to match in someCriteria.
        @code
        t_linklist* results = NULL;

        linklist_findall(myLinkList, &results, myCmpFunction, (void* )someCriteria);
        // do something here with the 'results' linklist
        // then free the results linklist
        linklist_chuck(results);
        @endcode

        @see	linklist_match
        @see	t_cmpfn
        @see	linklist_findfirst
    */
    void linklist_findall(t_linklist* x, t_linklist* *out, long cmpfn(void* , void* ), void* cmpdata);


    /**
        Call the named message on every object in the linklist.
        The linklist_methodall() function requires that all items in the linklist are
        object instances with a valid t_object header.

        @ingroup linklist
        @param	x	The linklist instance.
        @param	s	The name of the message to send to the objects.
        @param	...	Any arguments to be sent with the message.

        @remark		Internally, this function uses object_method(), meaning that no errors will be
                    posted if the message name does not exist for the object.  It also means that
                    messages sent methods with #A_GIMME definitions will need to be given a symbol
                    argument prior to the argc and argv array information.
    */
    void linklist_methodall(t_linklist* x, t_symbol* s, ...);


    /**
        Call the named message on an object specified by index.
        The item must be an object instance with a valid t_object header.

        @ingroup linklist
        @param	x	The linklist instance.
        @param	i	The index of the item to which to send the message.
        @param	s	The name of the message to send to the objects.
        @param	...	Any arguments to be sent with the message.

        @remark		Internally, this function uses object_method(), meaning that no errors will be
                    posted if the message name does not exist for the object.  It also means that
                    messages sent methods with #A_GIMME definitions will need to be given a symbol
                    argument prior to the argc and argv array information.
    */
    void* linklist_methodindex(t_linklist* x, t_atom_long i, t_symbol* s, ...);


    /**
        Sort the linked list.
        The items in the list are ordered using a #t_cmpfn function that is passed in as an argument.

        @ingroup linklist
        @param	x		The linklist instance.
        @param	cmpfn	The function used to sort the list.

        @remark		The following is example is a real-world example of sorting a linklist of symbols alphabetically
                    by first letter only.  First the cmpfn is defined, then it is used in a different function
                    by linklist_sort().
        @code
        long myAlphabeticalCmpfn(void* a, void* b)
        {
            t_symbol* s1 = (t_symbol* )a;
            t_symbol* s2 = (t_symbol* )b;

            if(s1->s_name[0] < s2->s_name[0])
                return true;
            else
                return false;
        }

        void mySortMethod(t_myobj *x)
        {
            // the linklist was already created and filled with items previously
            linklist_sort(x->myLinkList, myAlphabeticalCmpfn);
        }
        @endcode
    */
    void linklist_sort(t_linklist* x, long cmpfn(void* , void* ));


    /**
        Call the specified function for every item in the linklist.

        @ingroup linklist
        @param	x		The linklist instance.
        @param	fun		The function to call, specified as function pointer cast to a Max #method.
        @param	arg		An argument that you would like to pass to the function being called.

        @remark			The linklist_funall() method will call your function for every item in the list.
                        It will pass both a pointer to the item in the list, and any argument that you
                        provide.  The following example shows a function that could be called by linklist_funall().
        @code
        void myFun(t_object* myObj, void* myArg)
        {
            // do something with myObj and myArg here
            // myObj is the item in the linklist
        }
        @endcode
    */
    void linklist_funall(t_linklist* x, method fun, void* arg);


    /**
        Call the specified function for every item in the linklist.
        The iteration through the list will halt if the function returns a non-zero value.

        @ingroup linklist
        @param	x		The linklist instance.
        @param	fun		The function to call, specified as function pointer cast to a Max #method.
        @param	arg		An argument that you would like to pass to the function being called.

        @remark			The linklist_funall() method will call your function for every item in the list.
                        It will pass both a pointer to the item in the list, and any argument that you
                        provide.  The following example shows a function that could be called by linklist_funall().
        @code
        long myFun(t_symbol* myListItemSymbol, void* myArg)
        {
            // this function is called by a linklist that contains symbols for its items
            if(myListItemSymbol == gensym("")){
                error("empty symbol -- aborting linklist traversal")
                return 1;
            }
            else{
                // do something with the symbol
                return 0;
            }
        }
        @endcode
    */
    t_atom_long linklist_funall_break(t_linklist* x, method fun, void* arg);


    /**
        Call the specified function for an item specified by index.

        @ingroup linklist
        @param	x		The linklist instance.
        @param	i		The index of the item to which to send the message.
        @param	fun		The function to call, specified as function pointer cast to a Max #method.
        @param	arg		An argument that you would like to pass to the function being called.

        @remark			The linklist_funindex() method will call your function for an item in the list.
                        It will pass both a pointer to the item in the list, and any argument that you
                        provide.  The following example shows a function that could be called by linklist_funindex().
        @code
        void myFun(t_object* myObj, void* myArg)
        {
            // do something with myObj and myArg here
            // myObj is the item in the linklist
        }
        @endcode
    */
    void* linklist_funindex(t_linklist* x, long i, method fun, void* arg);


    /**
        Given an item in the list, replace it with a different value.

        @ingroup linklist
        @param	x		The linklist instance.
        @param	p		An item in the list.
        @param	newp	The new value.
        @return			Always returns NULL.
    */
    void* linklist_substitute(t_linklist* x, void* p, void* newp);


    /**
        Given an item in the list, find the next item.
        This provides an means for walking the list.

        @ingroup linklist
        @param	x		The linklist instance.
        @param	p		An item in the list.
        @param	next	The address of a pointer to set with the next item in the list.
    */
    void* linklist_next(t_linklist* x, void* p, void** next);


    /**
        Given an item in the list, find the previous item.
        This provides an means for walking the list.

        @ingroup linklist
        @param	x		The linklist instance.
        @param	p		An item in the list.
        @param	prev	The address of a pointer to set with the previous item in the list.
    */
    void* linklist_prev(t_linklist* x, void* p, void** prev);


    /**
        Return the last item (the tail) in the linked-list.

        @ingroup linklist
        @param	x		The linklist instance.
        @param	item	The address of pointer in which to store the last item in the linked-list.
        @return 		always returns NULL
    */
    void* linklist_last(t_linklist* x, void** item);


    /**
        Set the linklist's readonly bit.

        By default the readonly bit is 0, indicating that it is threadsafe for both reading and writing.
        Setting the readonly bit to 1 will disable the linklist's theadsafety mechanism, increasing
        performance but at the expense of threadsafe operation.
        Unless you can guarantee the threading context for a linklist's use, you should leave this set to 0.

        @ingroup linklist
        @param	x			The linklist instance.
        @param	readonly	A 1 or 0 for setting the readonly bit.
    */
    void linklist_readonly(t_linklist* x, long readonly);


    /**
        Set the linklist's datastore flags.  The available flags are enumerated in #e_max_datastore_flags.
        These flags control the behavior of the linklist, particularly when removing items from the list
        using functions such as linklist_clear(), linklist_deleteindex(), or when freeing the linklist itself.

        @ingroup linklist
        @param	x			The linklist instance.
        @param	flags	A valid value from the #e_max_datastore_flags.  The default is #OBJ_FLAG_OBJ.
    */
    void linklist_flags(t_linklist* x, long flags);


    /**
        Get the linklist's datastore flags.

        @ingroup linklist
        @param	x	The linklist instance.
        @return		The current state of the linklist flags as enumerated in #e_max_datastore_flags.
    */
    t_atom_long linklist_getflags(t_linklist* x);


    /**
        A linklist comparison method that determines if two item pointers are equal.

        @ingroup linklist

        @param	a		The first item to compare.
        @param	b		The second item to compare.
        @return			Returns 1 if the items are equal, otherwise 0.

        @see			t_cmpfn
    */
    long linklist_match(void* a, void* b);


    END_USING_C_LINKAGE

}} // namespace c74::max
