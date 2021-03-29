/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

    BEGIN_USING_C_LINKAGE


    /**	A low-level object for tempo-based scheduling.
        @ingroup	time
        @see		#t_timeobject
        @see		ITM */
    typedef t_object t_itm;


    // private -- internal use only
    typedef struct _clocksource
    {
        t_object c_ob;
        method c_getticks;		// returns the current tick count
        method c_getstate;		// returns 0 if transport not going, 1 if going
        t_symbol *c_name;		// name
        long c_usedcount;		// number of transports using this clock source
        method c_gettempo;		// get current tempo
        method c_gettimesig;	// get current timesig
        method c_getsr;			// get current samplerate
    } t_clocksource;

    // used by clocksource to report time sig change to ITM

    typedef struct _tschange {
        long c_num;
        long c_denom;
        double c_tsbaseticks;	// ticks at last ts change (use -1 for "unknown")
        long c_tsbasebars;		// bars at last ts change (use -1 for "unknown")
    } t_tschange;

    /**	Flags that determine attribute and time object behavior
        @ingroup time	*/
    enum {
        TIME_FLAGS_LOCATION = 1,			///< 1 1 0 location-based bar/beat/unit values (as opposed to interval values, which are 0 0 0 relative)
        TIME_FLAGS_TICKSONLY = 2,			///< only ticks-based values (not ms) are acceptable
        TIME_FLAGS_FIXEDONLY = 4,			///< only fixed values (ms, hz, samples) are acceptable
        TIME_FLAGS_LOOKAHEAD = 8,			///< add lookahead attribute (unsupported)
        TIME_FLAGS_USECLOCK = 16,			///< this time object will schedule events, not just hold a value
        TIME_FLAGS_USEQELEM = 32,			///< this time object will defer execution of scheduled events to low priority thread
        TIME_FLAGS_FIXED = 64,				///< will only use normal clock (i.e., will never execute out of ITM)
        TIME_FLAGS_PERMANENT = 128,			///< event will be scheduled in the permanent list (tied to a specific time)
        TIME_FLAGS_TRANSPORT = 256,			///< add a transport attribute
        TIME_FLAGS_EVENTLIST = 512,			///< add an eventlist attribute (unsupported)
        TIME_FLAGS_CHECKSCHEDULE = 1024,	///< internal use only
        TIME_FLAGS_LISTENTICKS = 2048,		///< flag for time_listen: only get notifications if the time object holds tempo-relative values
        TIME_FLAGS_NOUNITS = 4096,			///< internal use only
        TIME_FLAGS_BBUSOURCE = 8192,		///< source time was in bar/beat/unit values, need to recalculate when time sig changes
        TIME_FLAGS_POSITIVE = 16384			///< constrain any values < 0 to 0
    };


    /*******************************************************************************/

    /**	Return the global (default / unnamed) itm object.
        @ingroup	time
        @return		The global #t_itm object.	*/
    void *itm_getglobal(void);

    /**	Return a named itm object.
        @ingroup		time
        @param	s		The name of the itm to return.
        @param	scheduler	.
        @param	defaultclocksourcename	.
        @param	create	If non-zero, then create this named itm should it not already exist.
        @return			The global #t_itm object.	*/
    void *itm_getnamed(t_symbol *s, void *scheduler, t_symbol *defaultclocksourcename, long create);

    // currently the same as itm_getnamed(s,true);
    void *itm_getfromarg(t_object *o, t_symbol *s);

    /**	Reference an itm object.
        When you are using an itm object, you must call this function to increment its reference count.
        @ingroup	time
        @param	x	The itm object.	*/
    void itm_reference(t_itm *x);

    /**	Stop referencing an itm object.
        When you are done using an itm object, you must call this function to decrement its reference count.
        @ingroup	time
        @param	x	The itm object.	*/
    void itm_dereference(t_itm *x);


    // event list support is limited to use in javascript for the time being.
    void itm_deleteeventlist(t_itm *x, t_symbol *eventlist);
    void itm_eventlistseek(t_itm *x);
    void itm_geteventlistnames(t_itm *x, long *count, t_symbol ***names);
    void itm_switcheventlist(t_itm *x, t_symbol *eventlist, double offset);


    /**	Report the current internal time.
        This is the same as calling clock_getftime();
        @ingroup	time
        @param	x	The itm object.
        @return		The current internal time.	*/
    double itm_gettime(t_itm *x);

    /**	Report the current time of the itm in ticks.
        You can use functions such as itm_tickstobarbeatunits() or itm_tickstoms() to convert to a different representation of the time.
        @ingroup	time
        @param	x	The itm object.
        @return		The current time in ticks.	*/
    double itm_getticks(t_itm *x);

    /**	Print diagnostic information about an itm object to the Max window.
        @ingroup	time
        @param	x	The itm object.		*/
    void itm_dump(t_itm *x);


    // private -- internal use only
    void itm_sync(t_itm *x);



    /**	Set an itm object's current time signature.
        @ingroup		time
        @param	x		The itm object.
        @param	num		The top number of the time signature.
        @param	denom	The bottom number of the time signature.
        @param	flags	Currently unused -- pass zero.	*/
    void itm_settimesignature(t_itm *x, long num, long denom, long flags);

    /**	Query an itm object for its current time signature.
        @ingroup		time
        @param	x		The itm object.
        @param	num		The address of a variable to hold the top number of the time signature upon return.
        @param	denom	The address of a variable to hold the bottom number of the time signature upon return.	*/
    void itm_gettimesignature(t_itm *x, long *num, long *denom);

    void itm_seek(t_itm *x, double oldticks, double newticks, long chase);

    /**	Pause the passage of time for an itm object.
        This is the equivalent to setting the state of a transport object to 0 with a toggle.
        @ingroup		time
        @param	x		The itm object.		*/
    void itm_pause(t_itm *x);

    /**	Start the passage of time for an itm object, from it's current location.
        This is the equivalent to setting the state of a transport object to 0 with a toggle.
        @ingroup		time
        @param	x		The itm object.		*/
    void itm_resume(t_itm *x);

    /**	Find out if time is currently progressing for a given itm object.
        @ingroup		time
        @param	x		The itm object.
        @return			Returns non-zero if the time is running, or zero if it is paused.
        @see			itm_pause()
        @see			itm_resume()	*/
    long itm_getstate(t_itm *x);


    /**	Set the number of ticks-per-quarter-note globally for the itm system.
        The default is 480.
        @ingroup		time
        @param	res		The number of ticks-per-quarter-note.
        @see			itm_getresolution()	*/
    void itm_setresolution(double res);

    /**	Get the number of ticks-per-quarter-note globally from the itm system.
        @ingroup		time
        @return			The number of ticks-per-quarter-note.
        @see			itm_setresolution()	*/
    double itm_getresolution(void);


    /**	Given an itm object, return its name.
        @ingroup	time
        @param	x	The itm object.
        @return		The name of the itm.	*/
    t_symbol *itm_getname(t_itm *x);



    t_max_err itm_parse(t_itm *x, long argc, t_atom *argv, long flags, double *ticks, double *fixed, t_symbol **unit, long *bbu, char *bbusource);


    /**	Convert a time value in ticks to the equivalent value in milliseconds, given the context of a specified itm object.
        @ingroup		time
        @param	x		An itm object.
        @param	ticks	A time specified in ticks.
        @return			The time specified in ms.	*/
    double itm_tickstoms(t_itm *x, double ticks);

    /**	Convert a time value in milliseconds to the equivalent value in ticks, given the context of a specified itm object.
        @ingroup		time
        @param	x		An itm object.
        @param	ms		A time specified in ms.
        @return			The time specified in ticks.	*/
    double itm_mstoticks(t_itm *x, double ms);

    /**	Convert a time value in milliseconds to the equivalent value in samples, given the context of a specified itm object.
        @ingroup		time
        @param	x		An itm object.
        @param	ms		A time specified in ms.
        @return			The time specified in samples.	*/
    double itm_mstosamps(t_itm *x, double ms);

    /**	Convert a time value in samples to the equivalent value in milliseconds, given the context of a specified itm object.
        @ingroup		time
        @param	x		An itm object.
        @param	samps	A time specified in samples.
        @return			The time specified in ms.	*/
    double itm_sampstoms(t_itm *x, double samps);


    /**	Convert a time value in bbu to the equivalent value in ticks, given the context of a specified itm object.
        @ingroup			time
        @param	x			An itm object.
        @param	bars		The measure number of the location/position.
        @param	beats		The beat number of the location/position.
        @param	units		The number of ticks past the beat of the location/position.
        @param	ticks		The address of a variable to hold the number of ticks upon return.
        @param	position	Set this parameter to #TIME_FLAGS_LOCATION or to zero (for position mode).	*/
    void itm_barbeatunitstoticks(t_itm *x, long bars, long beats, double units, double *ticks, char position);

    /**	Convert a time value in bbu to the equivalent value in ticks, given the context of a specified itm object.
        @ingroup			time
        @param	x			An itm object.
        @param	ticks		The number of ticks to translate into a time represented as bars, beats, and ticks.
        @param	bars		The address of a variable to hold the measure number of the location/position upon return.
        @param	beats		The address of a variable to hold the beat number of the location/position upon return.
        @param	units		The address of a variable to hold the number of ticks past the beat of the location/position upon return.
        @param	position	Set this parameter to #TIME_FLAGS_LOCATION or to zero (for position mode).	*/
    void itm_tickstobarbeatunits(t_itm *x, double ticks, long *bars, long *beats, double *units, char position);


    void itm_format(t_itm *x, double ms, double ticks, long flags, t_symbol *unit, long *argc, t_atom **argv);


    /**	Given the name of a time unit (e.g. 'ms', 'ticks', 'bbu', 'samples', etc.), determine whether the unit is fixed
        (doesn't change with tempo, time-signature, etc.) or whether it is flexible.
        @ingroup		time
        @param	u		The name of the time unit.
        @return			Zero if the unit is fixed (milliseconds, for example) or non-zero if it is flexible (ticks, for example).	*/
    long itm_isunitfixed(t_symbol *u);



    void itmclock_delay(t_object *x, t_itm *m, t_symbol *eventlist, double delay, long quantization);
    void *itmclock_new(t_object *owner, t_object *timeobj, method task, method killer, long permanent);
    void itmclock_set(t_object *x, t_itm *m, t_symbol *eventlist, double time);
    void itmclock_unset(t_object *x);


    // private -- internal use only
    void *itm_clocksource_getnamed(t_symbol *name, long create);
    void itm_getclocksources(long *count, t_symbol ***sources);
    double itm_getsr(t_itm *x);
    double itm_gettempo(t_itm *x);


    /**	A high-level time object for tempo-based scheduling.

        @ingroup	time
        @see		#t_itm
        @see		ITM
    */
    typedef t_object t_timeobject;


    /*******************************************************************************/


    /** Stop a currently scheduled time object.
        @ingroup	time
        @param		x				The time object.
    */
    void time_stop(t_timeobject *x);


    /** Execute a time object's task, then if it was already set to execute, reschedule for the current interval value of the object.
        @ingroup	time
        @param		x				The time object.
    */
    void time_tick(t_timeobject *x);


    /** Convert the value of a time object to milliseconds.
        @ingroup	time
        @param		x				The time object.
        @return					The time object's value, converted to milliseconds.
    */
    double time_getms(t_timeobject *x);


    /** Convert the value of a time object to ticks.
        @ingroup	time
        @param		x				The time object.
        @return					The time object's value, converted to ticks.
    */
    double time_getticks(t_timeobject *x);


    /** Return the phase of the ITM object (transport) associated with a time object.
        @ingroup	time
        @param		tx				The time object.
        @param		phase			Pointer to a double to receive the progress within the specified time value of the associated ITM object.
        @param		slope			Pointer to a double to receive the slope (phase difference) within the specified time value of the associated ITM object.
        @param		ticks			Ticks
    */
    void time_getphase(t_timeobject *tx, double *phase, double *slope, double *ticks);


    /** Specify that a millisecond-based attribute to be updated automatically when the converted milliseconds of the time object's value changes.
        @ingroup	time
        @param		x				The time object.
        @param		attr			Name of the millisecond based attribute in the owning object that will be updated
        @param		flags			If TIME_FLAGS_LISTENTICKS is passed here, updating will not happen if the time value is fixed (ms) based
     */
    void time_listen(t_timeobject *x, t_symbol *attr, long flags);


    /** Set the current value of a time object (either an interval or a position) using a Max message.
        @ingroup	time
        @param		tx				The time object.
        @param		s				Message selector.
        @param		argc			Count of arguments.
        @param		argv			Message arguments.
     */
    void time_setvalue(t_timeobject *tx, t_symbol *s, long argc, t_atom *argv);

    /** Create an attribute permitting a time object to be changed in a user-friendly way.
        @ingroup	time
        @param		c				Class being initialized.
        @param		attrname		Name of the attribute associated with the time object.
        @param		attrlabel		Descriptive label for the attribute (appears in the inspector)
        @param		flags			Options, see "Flags that determine time object behavior" above
    */
    void class_time_addattr(t_class *c, const char *attrname, const char *attrlabel, long flags);

    /**	Create a new time object.
        @ingroup	time
        @param		owner			Object that will own this time object (task routine, if any, will pass owner as argument).
        @param		attrname		Name of the attribute associated with the time object.
        @param		tick			Task routine that will be executed (can be NULL)
        @param		flags			Options, see "Flags that determine time object behavior" above
        @return					The newly created #t_timeobject.
    */
    t_object *time_new(t_object *owner, t_symbol *attrname, method tick, long flags);

    /**	Create a new time object.
        @ingroup	time
        @param		owner			Object that will own this time object (task routine, if any, will pass baton as argument).
        @param		attrname		Name of the attribute associated with the time object.
        @param		tick			Task routine that will be executed (can be NULL)
        @param		flags			Options, see "Flags that determine time object behavior" above
        @param		baton			Passed as an argument to the task routine instead of owner as in time_new().
        @return					The newly created #t_timeobject.

        @since 7.2.5
     */
    t_object *time_new_custom(t_object *owner, t_symbol *attrname, method tick, long flags, void *baton);

    /**	Return a time object associated with an attribute of an owning object.
        @ingroup	time
        @param		owner			Object that owns this time object (task routine, if any, will pass owner as argument).
        @param		attrname		Name of the attribute associated with the time object.
        @return					The #t_timeobject associated with the named attribute.
    */
    t_object *time_getnamed(t_object *owner, t_symbol *attrname);


    void time_enable_attributes(t_object *x);

    /**	Return whether this time object currently holds a fixed (millisecond-based) value.
        @ingroup	time
        @param		x				Time object.
        @return					True if time object's current value is fixed, false if it is tempo-relative.
    */
    long time_isfixedunit(t_timeobject *x);


    /** Schedule a task, with optional quantization.
        @ingroup	time
        @param		x				The time object that schedules temporary events (must have been created with TIME_FLAGS_USECLOCK but not TIME_FLAGS_PERMANENT)
        @param		quantize		A time object that holds a quantization interval, can be NULL for no quantization.
    */
    void time_schedule(t_timeobject *x, t_timeobject *quantize);


    /** Schedule a task, with optional minimum interval.
        @ingroup	time
        @param		x				The time object that schedules temporary events (must have been created with TIME_FLAGS_USECLOCK but not TIME_FLAGS_PERMANENT)
        @param		quantize		The minimum interval into the future when the event can occur, can be NULL if there is no minimum interval.
    */
    void time_schedule_limit(t_timeobject *x, t_timeobject *quantize);

    /** Schedule a task for right now, with optional quantization.
        @ingroup	time
        @param		x				The time object that schedules temporary events. The time interval is ignored and 0 ticks is used instead.
        @param		quantize		A time object that holds a quantization interval, can be NULL for no quantization.
    */
    void time_now(t_timeobject *x, t_timeobject *quantize);


    /** Return the ITM object associated with this time object.
        @ingroup	time
        @param		ox				Time object.
        @return					The associated #t_itm object.
    */
    void *time_getitm(t_timeobject *ox);


    /** Calculate the quantized interval (in ticks) if this time object were to be scheduled at the current time.
        @ingroup	time
        @param		ox				Time object.
        @param		vitm			The associated ITM object (use time_getitm() to determine it).
        @param		oq				A time object that holds a quantization interval, can be NULL.
        @return					Interval (in ticks) for scheduling this object.
    */
    double time_calcquantize(t_timeobject *ox, t_itm *vitm, t_timeobject *oq);


    /** Associate a named setclock object with a time object (unsupported).
        @ingroup	time
        @param		tx				Time object.
        @param		sc				Name of an associated setclock object.
    */
    void time_setclock(t_timeobject *tx, t_symbol *sc);


    END_USING_C_LINKAGE

}} // namespace c74::max
