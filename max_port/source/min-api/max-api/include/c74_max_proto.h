/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {


    BEGIN_USING_C_LINKAGE

    /**
        Release the memory used by a Max object.
        freeobject() calls an object's free function, if any, then disposes the
        memory used by the object itself. freeobject() should be used on any
        instance of a standard Max object data structure, with the exception of
        Qelems and Atombufs. Clocks, Binbufs, Proxies, Exprs, etc. should be freed with freeobject().

        @ingroup	class_old
        @param		op	The object instance pointer to free.

        @remark		This function can be replaced by the use of object_free().
        Unlike freeobject(), object_free() checkes to make sure the pointer is
        not NULL before trying to free it.

        @see newobject()
        @see object_free()
    */
    void freeobject(t_object *op);


    /** Make a new instance of an existing Max class.
        @ingroup class_old

        @param s	className Symbol specifying the name of the class of the instance to be created.
        @param argc	Count of arguments in argv.
        @param argv	Array of t_atoms; arguments to the class's instance creation function.

        @return 	A pointer to the created object, or 0 if the class
        didn't exist or there was another type of error in creating the instance.

        @remark This function creates a new instance of the specified class. Using
        newinstance is equivalent to typing something in a New Object box
        when using Max. The difference is that no object box is created in any
        Patcher window, and you can send messages to the object directly
        without connecting any patch cords. The messages can either be type-
        checked (using typedmess) or non-type-checked (using the members
        of the getfn family).

        This function is useful for taking advantage of other already-defined
        objects that you would like to use 'privately' in your object, such as
        tables. See the source code for the coll object for an example of using a
        privately defined class.
    */
    void *newinstance(const t_symbol *s, short argc, const t_atom *argv);


    /**	Use the setup() function to initialize your class by informing Max of its size,
        the name of your functions that create and destroy instances,
        and the types of arguments passed to the instance creation function.

        @ingroup class_old

        @param	ident		A global variable in your code that points to the initialized class.
        @param	makefun		Your instance creation function.
        @param	freefun		Your instance free function (see Chapter 7).
        @param	size		The size of your objects data structure in bytes.
                            Usually you use the C sizeof operator here.
        @param	menufun		No longer used.  You should pass NULL for this parameter.
        @param	type		The first of a list of arguments passed to makefun when an object is created.
        @param	...			Any additional arguments passed to makefun when an object is created.
                            Together with the type parameter, this creates a standard Max type list
                            as enumerated in #e_max_atomtypes.
                            The final argument of the type list should be a 0.
        @see @ref chapter_anatomy
    */
    C74_DEPRECATED ( void setup(t_messlist **ident, method makefun, method freefun, t_ptr_uint size, method menufun, short type, ...) )	;


    /**	Use addmess() to bind a function to a message other than the standard ones
        covered by addbang(), addint(), etc.

        @ingroup class_old
        @param	f 		Function you want to be the method.
        @param	s 		C string defining the message.
        @param	type	The first of one or more integers from #e_max_atomtypes specifying the arguments to the message.
        @param	...		Any additional types from #e_max_atomtypes for additonal arguments.
        @see @ref chapter_anatomy
    */
    C74_DEPRECATED ( void addmess(method f, const char* s, short type, ...) );


    /**	Used to bind a function to the common triggering message bang.

        @ingroup class_old
        @param	f	Function to be the bang method.
    */
    C74_DEPRECATED ( void addbang(method f) );


    /**
        Use addint() to bind a function to the int message received in the leftmost inlet.
        @ingroup class_old
        @param	f Function to be the int method.
    */
    C74_DEPRECATED ( void addint(method f) );


    /**	Use addfloat() to bind a function to the float message received in the leftmost inlet.
        @ingroup class_old
        @param	f Function to be the int method.
    */
    C74_DEPRECATED ( void addfloat(method f) );


    /**	Use addinx() to bind a function to a int message that will be received in
            an inlet other than the leftmost one.

        @ingroup class_old
        @param	f	Function to be the int method.
         @param	n	Number of the inlet connected to this method.
                    1 is the first inlet to the right of the left inlet.

        @remark	This correspondence between inlet locations and messages is not
                automatic, but it is strongly suggested that you follow existing practice.
                You must set the correspondence up when creating an object of your
                class with proper use of intin and floatin in your instance creation
                function @ref chapter_anatomy_object_new.
    */
    C74_DEPRECATED ( void addinx(method f, short n) );


    /**	Use addftx() to bind a function to a float message that will be received in
            an inlet other than the leftmost one.

        @ingroup class_old
        @param	f	Function to be the float method.
        @param	n	Number of the inlet connected to this method.
                    1 is the first inlet to the right of the left inlet.

        @remark	This correspondence between inlet locations and messages is not
                automatic, but it is strongly suggested that you follow existing practice.
                You must set the correspondence up when creating an object of your
                class with proper use of intin and floatin in your instance creation
                function @ref chapter_anatomy_object_new.
    */
    C74_DEPRECATED ( void addftx(method f, short n) );


    /**	Use newobject to allocate the space for an instance of your class and
        initialize its object header.

        @ingroup class_old
        @param	maxclass	The global class variable initialized in your main routine by the setup function.
        @return 			A pointer to the new instance.

        @remark				You call newobject() when creating an instance of your class in your
                            creation function. newobject allocates the proper amount of memory
                            for an object of your class and installs a pointer to your class in the
                            object, so that it can respond with your class's methods if it receives a
                            message.
    */
    C74_DEPRECATED ( void* newobject(void* maxclass) );


    /**	Given a C-string, fetch the matching #t_symbol pointer from the symbol table,
        generating the symbol if neccessary.

        @ingroup symbol
        @param	s	A C-string to be looked up in Max's symbol table.
        @return		A pointer to the #t_symbol in the symbol table.
    */
    t_symbol* gensym(const char* s);
    t_symbol* gensym_tr(const char* s);
    char *str_tr(const char *s);


    /**	Print text to the system console.
        On the Mac this post will be visible by launching Console.app in the /Applications/Utilities folder.
        On Windows this post will be visible by launching the dbgView.exe program, which is a free download
        as a part of Microsoft's SysInternals.

        @ingroup console
        @param	fmt		A C-string containing text and printf-like codes
                        specifying the sizes and formatting of the additional arguments.
        @param	...		Arguments of any type that correspond to the format codes in fmtString.

        @remark			Particularly on MacOS 10.5, posting to Console.app can be a computationally expensive operation.
                        Use with care.

        @see object_post()
    */
    void cpost(const char* fmt, ...);


    /**	Receive messages from the error handler.
        @ingroup misc
        @param	x	The object to be subscribed to the error handler.

        @remark		error_subscribe() enables your object to receive a message (error),
                    followed by the list of atoms in the error message posted to the Max
                    window.

                    Prior to calling error_subscribe(), you should bind the error
                    message to an internal error handling routine:
        @code
        addmess((method)myobject_error, "error", A_GIMME, 0);
        @endcode
                    Your error handling routine should be declared as follows:
        @code
        void myobject_error(t_myobject *x, t_symbol* s, short argc, t_atom* argv);
        @endcode
    */
    void error_subscribe(t_object* x);


    /** Remove an object as an error message recipient.
        @ingroup misc
        @param	x	The object to unsubscribe.
    */
    void error_unsubscribe(t_object* x);


    /**	Print text to the Max window, linked to an instance of your object.

        Max window rows which are generated using object_post() or object_error() can be
        double-clicked by the user to have Max assist with locating the object in a patcher.
        Rows created with object_post() and object_error() will also automatically provide
        the name of the object's class in the correct column in the Max window.

        @ingroup console
        @param	x		A pointer to your object.
        @param	s		A C-string containing text and printf-like codes
                        specifying the sizes and formatting of the additional arguments.
        @param	...		Arguments of any type that correspond to the format codes in fmtString.

        @remark			Example:
        @code
        void myMethod(myObject *x, long someArgument)
        {
            object_post((t_object*)x, "This is my argument: %ld", someArgument);
        }
        @endcode

        @see object_error()
    */
    void object_post(const t_object* x, const char* s, ...);


    /**	Print text to the Max window, linked to an instance of your object,
        and flagged as an error (highlighted with a red background).

        Max window rows which are generated using object_post() or object_error() can be
        double-clicked by the user to have Max assist with locating the object in a patcher.
        Rows created with object_post() and object_error() will also automatically provide
        the name of the object's class in the correct column in the Max window.

        @ingroup console
        @param	x		A pointer to your object.
        @param	s		A C-string containing text and printf-like codes
                        specifying the sizes and formatting of the additional arguments.
        @param	...		Arguments of any type that correspond to the format codes in fmtString.

        @see object_post()
        @see object_warn()
    */
    void object_error(const t_object* x, const char* s, ...);


    /**	Print text to the Max window, linked to an instance of your object,
        and flagged as a warning (highlighted with a yellow background).

        Max window rows which are generated using object_post(), object_error(), or object_warn can be
        double-clicked by the user to have Max assist with locating the object in a patcher.
        Rows created with object_post(), object_error(), or object_warn() will also automatically provide
        the name of the object's class in the correct column in the Max window.

        @ingroup console
        @param	x		A pointer to your object.
        @param	s		A C-string containing text and printf-like codes
                        specifying the sizes and formatting of the additional arguments.
        @param	...		Arguments of any type that correspond to the format codes in fmtString.

        @see object_post()
        @see object_error()
    */
    void object_warn(const t_object* x, const char* s, ...);


    /**	Print text to the Max window, linked to an instance of your object,
        and flagged as an error (highlighted with a red background),
        and grab the user's attention by displaying a banner in the patcher window.

        This function should be used exceedingly sparingly, with preference given to
        object_error() when a problem occurs.

        @ingroup console
        @param	x		A pointer to your object.
        @param	s		A C-string containing text and printf-like codes
                        specifying the sizes and formatting of the additional arguments.
        @param	...		Arguments of any type that correspond to the format codes in fmtString.

        @see object_post()
        @see object_error()
    */
    void object_error_obtrusive(const t_object* x, const char* s, ...);


    // inlet/outlet functions

    /**	Use inlet_new() to create an inlet that can receive a specific message or any message.

        @ingroup inout
        @param	x	Your object.
        @param	s	Character string of the message, or NULL to receive any message.
        @return		A pointer to the new inlet.

        @remark		inlet_new() ceates a general purpose inlet.
                    You can use it in circumstances where you would like special messages to be received in
                    inlets other than the leftmost one.
                    To create an inlet that receives a particular message, pass the message's
                    character string. For example, to create an inlet that receives only bang
                    messages, do the following
        @code
        inlet_new (myObject,"bang");
        @endcode

        @remark		To create an inlet that can receive any message, pass NULL for msg
        @code
        inlet_new (myObject, NULL);
        @endcode

        @remark		Proxies are an alternative method for general-purpose inlets that have
                    a number of advantages. If you create multiple inlets as shown above,
                    there would be no way to figure out which inlet received a message. See
                    the discussion in @ref chapter_inout_proxies.
    */
    void* inlet_new(void* x, const char* s);


    // for dynamic inlets
    void* inlet_append(t_object* op, void* who, t_symbol* s1, t_symbol* s2);
    void inlet_delete(void* x);
    void* inlet_nth(t_object* x, long n);
    long inlet_count(t_object* x);


    /**	Use outlet_new() to create an outlet that can send a specific non-standard message, or any message.

        @ingroup	inout
        @param	x	Your object.
        @param	s	A C-string specifying the message that will be sent out this outlet,
                    or NULL to indicate the outlet will be used to send various messages.
                    The advantage of this kind of outlet's flexibility is balanced by the fact that
                    Max must perform a message-lookup in real-time for every message sent through it,
                    rather than when a patch is being constructed, as is true for other types of outlets.
                    Patchers execute faster when outlets are typed, since the message
                    lookup can be done before the program executes.
        @return		A pointer to the new outlet.
    */
    void* outlet_new(void* x, const char* s);


    // for dynamic outlets
    void* outlet_append(t_object* op, t_symbol* s1, t_symbol* s2);
    void* outlet_insert_after(t_object* op, t_symbol* s1, t_symbol* s2, void* previous_outlet);
    void outlet_delete(void* x);
    long outlet_count(t_object* x);
    void* outlet_nth(t_object* x, long n);


    /**	Use outlet_bang() to send a bang message out an outlet.

        @ingroup inout
        @param	o	Outlet that will send the message.
        @return		Returns 0 if a stack overflow occurred, otherwise returns 1.
    */
    void* outlet_bang(void* o);


    /**	Use outlet_int() to send an int message out an outlet.

        @ingroup inout
        @param	o	Outlet that will send the message.
        @param	n	Integer value to send.
        @return		Returns 0 if a stack overflow occurred, otherwise returns 1.
    */
    void* outlet_int(void* o, t_atom_long n);


    /**	Use outlet_float() to send a float message out an outlet.

        @ingroup inout
        @param	o	Outlet that will send the message.
        @param	f	Float value to send.
        @return		Returns 0 if a stack overflow occurred, otherwise returns 1.
    */
    void* outlet_float(void* o, double f);



    /**	Use outlet_list() to send a list message out an outlet.

        @ingroup inout
        @param	o		Outlet that will send the message.
        @param	s		Should be NULL, but can be the _sym_list.
        @param	ac		Number of elements in the list in argv.
        @param	av		Atoms constituting the list.
        @return			Returns 0 if a stack overflow occurred, otherwise returns 1.

        @remark			outlet_list() sends the list specified by argv and argc out the
                        specified outlet. The outlet must have been created with listout or
                        outlet_new in your object creation function (see above). You create
                        the list as an array of Atoms, but the first item in the list must be an
                        integer or float.

                        Here's an example of sending a list of three numbers.
        @code
        t_atom myList[3];
        long theNumbers[3];
        short i;

        theNumbers[0] = 23;
        theNumbers[1] = 12;
        theNumbers[2] = 5;
        for (i=0; i < 3; i++) {
            atom_setlong(myList+i,theNumbers[i]);
        }
        outlet_list(myOutlet,0L,3,&myList);
        @endcode

        @remark			It's not a good idea to pass large lists to outlet_list that are
                        comprised of local (automatic) variables. If the list is small, as in the
                        above example, there's no problem. If your object will regularly send
                        lists, it might make sense to keep an array of t_atoms inside your
                        object's data structure.
    */
    void* outlet_list(void* o, t_symbol* s, short ac, const t_atom* av);


    /**	Use outlet_anything() to send any message out an outlet.

        @ingroup inout
        @param	o		Outlet that will send the message.
        @param	s		The message selector #t_symbol*.
        @param	ac		Number of elements in the list in argv.
        @param	av		Atoms constituting the list.
        @return			Returns 0 if a stack overflow occurred, otherwise returns 1.

        @remark			This function lets you send an arbitrary message out an outlet.
                        Here are a couple of examples of its use.

                        First, here's a hard way to send the bang message (see outlet_bang() for an easier way):
        @code
        outlet_anything(myOutlet, gensym("bang"), 0, NIL);
        @endcode

        @remark			And here's an even harder way to send a single integer (instead of using outlet_int()).
        @code
        t_atom myNumber;

        atom_setlong(&myNumber, 432);
        outlet_anything(myOutlet, gensym("int"), 1, &myNumber);
        @endcode

        @remark			Notice that outlet_anything() expects the message argument as a
                        #t_symbol*, so you must use gensym() on a character string.

                        If you'll be sending the same message a lot, you might call gensym() on the message string at
                        initialization time and store the result in a global variable to save the
                        (significant) overhead of calling gensym() every time you want to send a
                        message.

                        Also, do not send lists using outlet_anything() with list as
                        the selector argument. Use the outlet_list() function instead.
    */
    void* outlet_anything(void* o, const t_symbol* s, short ac, const t_atom* av);


    // clock functions

    /**	Create a new Clock object.
        Normally, clock_new() is called in your instance creation
        function—and it cannot be called from a thread other than the main thread.
        To get rid of a clock object you created, use freeobject().

        @ingroup clocks
        @param	obj		Argument that will be passed to clock function fn when it is called.
                        This will almost always be a pointer to your object.
        @param	fn		Function to be called when the clock goes off,
                        declared to take a single argument as shown in @ref clocks_using_clocks.
        @return			A pointer to a newly created Clock object.
    */
    t_clock* clock_new(void* obj, method fn);


    /**	Cancel the scheduled execution of a Clock.
        clock_unset() will do nothing (and not complain) if the Clock passed
        to it has not been set.

        @ingroup clocks
        @param	x		Clock to cancel.
    */
    void clock_unset(void* x);


    /**	Schedule the execution of a Clock using a floating-point argument.
        clock_delay() sets a clock to go off at a certain number of
        milliseconds from the current logical time.

        @ingroup clocks
        @param	c		Clock to schedule.
        @param	time	Delay, in milliseconds, before the Clock will execute.
    */
    void clock_fdelay(void* c, double time);


    /**	Find out the current logical time of the scheduler in milliseconds
        as a floating-point number.

        @ingroup clocks
        @param	time	Returns the current time.
        @see	gettime()
        @see	setclock_getftime()
        @see	setclock_gettime()
    */
    void clock_getftime(double* time);


    // real-time

    /**	While most Max timing references "logical" time derived from Max's millisecond scheduler,
        time values produced by the systimer_gettime() are referenced from the CPU clock and can be used
        to time real world events with microsecond precision.

        The standard 'cpuclock' external in Max is a simple wrapper around this function.

        @ingroup	clocks
        @return		Returns the current real-world time.
    */
    double systimer_gettime(void);


    // scheduler functions

    /**	Find out the current logical time of the scheduler in milliseconds.

        @ingroup	clocks
        @return		Returns the current time.
        @see		clock_getftime()
    */
    long gettime(void);


    /**	Find the correct scheduler for the object and return the current time in milliseconds.

        @ingroup	clocks
        @return		Returns the current time.
        @see		clock_getftime()
     */
    double gettime_forobject(t_object* x);


    // queue functions

    /**	Create a new Qelem.
        The created Qelem will need to be freed using qelem_free(), do not use freeobject().

        @ingroup qelems
        @param	obj	Argument to be passed to function fun when the Qelem executes.
                    Normally a pointer to your object.
        @param	fn	Function to execute.
        @return		A pointer to a Qelem instance.
                    You need to store this value to pass to qelem_set().

        @remark		Any kind of drawing or calling of Macintosh Toolbox routines that
                    allocate or purge memory should be done in a Qelem function.
    */
    t_qelem* qelem_new(void* obj, method fn);


    /**	Cause a Qelem to execute.

        @ingroup qelems
        @param	q	The Qelem whose function will be executed in the main thread.

        @remark		The key behavior of qelem_set() is this: if the Qelem object has already
                    been set, it will not be set again. (If this is not what you want, see
                    defer().) This is useful if you want to redraw the state of some
                    data when it changes, but not in response to changes that occur faster
                    than can be drawn. A Qelem object is unset after its queue function has
                    been called.
    */
    void qelem_set(t_qelem* q);


    /**	Cancel a Qelem's execution.
        If the Qelem's function is set to be called, qelem_unset() will stop it
        from being called. Otherwise, qelem_unset() does nothing.

        @ingroup qelems
        @param	q	The Qelem whose execution you wish to cancel.
    */
    void qelem_unset(t_qelem* q);


    /**	Free a Qelem object created with qelem_new().
         Typically this will be in your object's free funtion.

        @ingroup qelems
        @param	x	The Qelem to destroy.
    */
    void qelem_free(t_qelem* x);


    /**	Cause a Qelem to execute with a higher priority.
        This function is identical to qelem_set(), except that the Qelem's
        function is placed at the front of the list of routines to execute in the
        main thread instead of the back. Be polite and only use
        qelem_front() only for special time-critical applications.

        @ingroup qelems
        @param	x	The Qelem whose function will be executed in the main thread.
    */
    void qelem_front(t_qelem* x);


    /**	Defer execution of a function to the main thread if (and only if)
        your function is executing in the scheduler thread.

        @ingroup	threading
        @param		ob		First argument passed to the function fun when it executes.
        @param		fn		Function to be called, see below for how it should be declared.
        @param		sym		Second argument passed to the function fun when it executes.
        @param		argc	Count of arguments in argv. argc is also the third argument passed to the function fun when it executes.
        @param		argv	Array containing a variable number of #t_atom function arguments.
                            If this argument is non-zero, defer allocates memory to make a copy of the arguments
                            (according to the size passed in argc)
                            and passes the copied array to the function fun when it executes as the fourth argument.
        @return		Return values is for internal Cycling '74 use only.

        @remark		This function uses the isr() routine to determine whether you're at the
                    Max timer interrupt level (in the scheduler thread).
                    If so, defer() creates a Qelem (see @ref qelems), calls
                    qelem_front(), and its queue function calls the function fn you
                    passed with the specified arguments.
                    If you're not in the scheduler thread, the function is executed immediately with the
                    arguments. Note that this implies that defer() is not appropriate for
                    using in situations such as Device or File manager I/0 completion routines.
                    The defer_low() function is appropriate however, because it always defers.

                    The deferred function should be declared as follows:
        @code
        void myobject_do (myObject *client, t_symbol* s, short argc, t_atom* argv);
        @endcode

        @see		defer_low()
    */
    void* defer(void* ob,method fn,t_symbol* sym,short argc,t_atom* argv);


    /**	Defer execution of a function to the back of the queue on the main thread.

        @ingroup	threading
        @param		ob		First argument passed to the function fun when it executes.
        @param		fn		Function to be called, see below for how it should be declared.
        @param		sym		Second argument passed to the function fun when it executes.
        @param		argc	Count of arguments in argv. argc is also the third argument passed to the function fun when it executes.
        @param		argv	Array containing a variable number of #t_atom function arguments.
                            If this argument is non-zero, defer allocates memory to make a copy of the arguments
                            (according to the size passed in argc)
                            and passes the copied array to the function fun when it executes as the fourth argument.
        @return		Return values is for internal Cycling '74 use only.

        @remark		defer_low() always defers a call to the function fun whether you are already
                    in the main thread or not, and uses qelem_set(), not qelem_front(). This
                    function is recommended for responding to messages that will cause
                    your object to open a dialog box, such as read and write.

                    The deferred function should be declared as follows:
        @code
        void myobject_do (myObject *client, t_symbol* s, short argc, t_atom* argv);
        @endcode

        @see		defer()
    */
    void* defer_low(void* ob,method fn,t_symbol* sym,short argc,t_atom* argv);

    void* defer_medium(void* ob, method fn, t_symbol* sym, short argc, t_atom* argv);

    void* defer_front(void* ob, method fn, t_symbol* sym, short argc, t_atom* argv);


    // message functions

    /** Use zgetfn() to send an untyped message to a Max object without error checking.
        @ingroup class_old

        @param op	Receiver of the message.
        @param msg	Message selector.
        @return		zgetfn returns a pointer to the method bound to the message selector
                    msg in the receiver's message list. It returns 0 but doesn't print an
                    error message in Max Window if the method can't be found.
    */
    method zgetfn(t_object* op, t_symbol* msg);


    // proxy functions

    /**	Use proxy_new to create a new Proxy object.

        @ingroup inout
        @param	x			Your object.
        @param	id			A non-zero number to be written into your object when a message is received in this particular Proxy.
                            Normally, id will be the inlet number analogous to in1, in2 etc.
        @param	stuffloc	A pointer to a location where the id value will be written.
        @return				A pointer to the new proxy inlet.

        @remark		This routine creates a new Proxy object (that includes an inlet). It
                    allows you to identify messages based on an id value stored in the
                    location specified by stuffLoc. You should store the pointer
                    returned by proxy_new() because you'll need to free all Proxies in your
                    object's free function using object_free().

                    After your method has finished, Proxy sets the stuffLoc location
                    back to 0, since it never sees messages coming in an object's leftmost
                    inlet. You'll know you received a message in the leftmost inlet if the
                    contents of stuffLoc is 0. As of Max 4.3, stuffLoc is not always
                    guaranteed to be a correct indicator of the inlet in which a message was
                    received. Use proxy_getinlet() to determine the inlet number.
    */
    void* proxy_new(void* x, long id, long* stuffloc);


    /**	Use proxy_getinlet to get the inlet number in which a message was received.
        Note that the <code>owner</code> argument should point to your external object's instance, not a proxy object.

        @ingroup inout
        @param	master	Your object.
        @return			The index number of the inlet that received the message.
    */
    long proxy_getinlet(t_object* master);


    // quittask functions

    /**	Register a function that will be called when Max exits.

        @ingroup misc
        @param	m	A function that will be called on Max exit.
        @param	a	Argument to be used with method m.

        @remark		quittask_install() provides a mechanism for your external to
        register a routine to be called prior to Max shutdown. This is useful for
        objects that need to provide disk-based persistance outside the
        standard Max storage mechanisms, or need to shut down hardware or
        their connection to system software and cannot do so in the
        termination routine of a code fragment.
    */
    void quittask_install(method m, void* a);


    /**	Unregister a function previously registered with quittask_install().

        @ingroup misc
        @param	m	Function to be removed as a shutdown method.
    */
    void quittask_remove(method m);


    // miscellaneous functions

    /**	Determine version information about the current Max environment.

        This function returns the version number of Max. In Max versions
        2.1.4 and later, this number is the version number of the Max kernel
        application in binary-coded decimal. Thus, 2.1.4 would return 214 hex
        or 532 decimal. Version 3.0 returns 300 hex.

        Use this to check for the existence of particular function macros that are only present in more
        recent Max versions. Versions before 2.1.4 returned 1, except for
        versions 2.1.1 - 2.1.3 which returned 2.

        Bit 14 (counting from left) will
        be set if Max is running as a standalone application, so you should
        mask the lower 12 bits to get the version number.

        @ingroup	misc
        @return		The Max environment's version number.
    */
    short maxversion(void);


    /**	Use open_promptset() to add a prompt message to the open file dialog displayed by open_dialog().

        Calling this function before open_dialog() permits a string to
        displayed in the dialog box instructing the user as to the purpose of the
        file being opened. It will only apply to the call of open_dialog() that
        immediately follows open_promptset().

        @ingroup files
        @param	s		A C-string containing the prompt you wish to display in the dialog box.

        @see open_dialog()
    */
    void open_promptset(const char* s);


    /**	Use saveas_promptset() to add a prompt message to the open file dialog displayed by saveas_dialog()
        or saveasdialog_extended().

        Calling this function before saveasdialog_extended() permits a string to
        displayed in the dialog box instructing the user as to the purpose of the
        file being opened. It will only apply to the call of saveasdialog_extended() that
        immediately follows saveas_promptset().

        @ingroup files
        @param	s		A C-string containing the prompt you wish to display in the dialog box.

        @see open_dialog()
    */
    void saveas_promptset(const char* s);


    // filewatch functions

    /**	Create a new filewatcher.
        The file will not be actively watched until filewatcher_start() is called.
        The filewatcher can be freed using object_free().

        @ingroup			files
        @param	owner		Your object.
                            This object will receive the message "filechanged" when the watcher sees a change in the file or folder.
        @param	path		The path in which the file being watched resides, or the path of the folder being watched.
        @param	filename	The name of the file being watched, or an empty string if you are simply watching the folder specified by path.
        @return				A pointer to the new filewatcher.
        @remark				The "filechanged" method should have the prototype:
        @code
        void myObject_filechanged(t_myObject *x, char* filename, short path);
        @endcode
    */
    void* filewatcher_new(t_object* owner, const short path, const char* filename);


    /**	Start watching a file using a filewatcher created with filewatcher_new().
        @param	x			A filewatcher pointer, as returned by filewatcher_new().
    */
    void filewatcher_start(void* x);


    /**	Stop watching a file using a filewatcher created with filewatcher_new().
        @param	x			A filewatcher pointer, as returned by filewatcher_new().
    */
    void filewatcher_stop(void* x);



    // fileusage functions

    /**	Add a file to a collective.
        @ingroup		files
        @param	w		Handle for the collective builder.
        @param	flags	If flags == 1, copy this file to support folder of an app instead of to the collective in an app.
        @param	name	The name of the file.
        @param	path	The path of the file to add.
    */
    void fileusage_addfile(void* w, long flags, const char* name, const short path);

    void fileusage_addfilename(void* w, long flags, const char* name);

    /**	Add a package to a standalone.
        @ingroup					files
        @param	w					Handle for the standalone builder
        @param	name				The name of the package
        @param	subfoldernames		A #t_atomarray containing symbols, each of which is a foldername in the package to include.
                                    Pass NULL to include the entire package contents.
                                    DO NOT FREE subfoldernames after passing it! Max takes over ownership of this atomarray!
        @version					Introduced in Max 7.0.4
    */
    void fileusage_addpackage(void *w, const char *name, t_object *subfoldernames);

    void fileusage_addpathname(void* w, long flags, const char* name);
    void fileusage_copyfolder(void* w, const char* name, long recursive);
    void fileusage_makefolder(void* w, const char* name);


    /** Present the user with the standard open file dialog.
        This function is convenient wrapper for using Mac OS Navigation
        Services or Standard File for opening files.

        The mapping of extensions to types is configured in the C74:/init/max-fileformats.txt file.
        The standard types to use for Max files are 'maxb' for old-format binary files,
        'TEXT' for text files, and 'JSON' for newer format patchers or other .json files.

        @ingroup files
        @param	name	A C-string that will receive the name of the file the user wants to open.
        The C-string should be allocated with a size of at least #MAX_FILENAME_CHARS.
        @param	volptr	Receives the Path ID of the file the user wants to open.
        @param	typeptr	The file type of the file the user wants to open.
        @param	types	A list of file types to display. This is not limited to 4
        types as in the SFGetFile() trap. Pass NULL to display all types.
        @param	ntypes	The number of file types in typelist. Pass 0 to display all types.

        @return			0 if the user clicked Open in the dialog box.
        If the user cancelled, open_dialog() returns a non-zero value.

        @see saveasdialog_extended()
        @see locatefile_extended()
     */
    short open_dialog(char* name, short* volptr, t_fourcc* typeptr, t_fourcc* types, short ntypes);


    /** Present the user with the standard save file dialog with your own list of file types.

        saveasdialog_extended() is similar to saveas_dialog(), but allows the
        additional feature of specifying a list of possible types. These will be
        displayed in a pop-up menu.

        File types found in the typelist argument that match known Max types
        will be displayed with descriptive text. Unmatched types will simply
        display the type name (for example, "foXx" is not a standard type so it
        would be shown in the pop-up menu as foXx)

        Known file types include:
        - TEXT: text file
        - maxb: Max binary patcher
        - maxc: Max collective
        - Midi: MIDI file
        - Sd2f: Sound Designer II audio file
        - NxTS: NeXT/Sun audio file
        - WAVE: WAVE audio file.
        - AIFF: AIFF audio file
        - mP3f: Max preference file
        - PICT: PICT graphic file
        - MooV: Quicktime movie file
        - aPcs: VST plug-in
        - AFxP: VST effect patch data file
        - AFxB: VST effect bank data file
        - DATA: Raw data audio file
        - ULAW: NeXT/Sun audio file

        @ingroup files
        @param	name		A C-string containing a default name for the file to save.
                            If the user decides to save a file, its name is returned here.
                            The C-string should be allocated with a size of at least #MAX_FILENAME_CHARS.

        @param	vol			If the user decides to save the file, the Path ID of the location chosen is returned here.

        @param	type		Returns the type of file chosen by the user.
        @param	typelist	The list of types provided to the user.
        @param	numtypes	The number of file types in typelist.

        @return				0 if the user choose to save the file.
                            If the user cancelled, returns a non-zero value.

        @see open_dialog()
        @see locatefile_extended()
     */
    short saveasdialog_extended(char* name, short* vol, t_fourcc* type, t_fourcc* typelist, short numtypes);

    void saveas_autoextension(char way);
    void saveas_setselectedtype(t_fourcc type);

    END_USING_C_LINKAGE

}} // namespace c74::max
