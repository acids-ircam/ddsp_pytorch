/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_max.h"

namespace c74 {
namespace max {

    /**	MSP System Properties.
        @ingroup	 msp	*/
    enum {
        SYS_MAXBLKSIZE = 2048,	///< a good number for a maximum signal vector size
        SYS_MAXSIGS = 250		///< number of signal inlets you can have in an object
    };


    // z_misc flags:
    static const int Z_NO_INPLACE = 1;		///< flag indicating the object doesn't want signals in place @ingroup msp
    static const int Z_PUT_LAST = 2;		///< when list of ugens is resorted, put this object at end @ingroup msp
    static const int Z_PUT_FIRST = 4;		///< when list of ugens is resorted, put this object at beginning @ingroup msp
    static const int Z_IGNORE_DISABLE = 8;	///< ignore the disable field, e.g. to process the pass~ object in a muted patcher.
    static const int Z_DONT_ADD = 16;		///< ignore this object -- its dsp method won't be called.
    static const int Z_MC_INLETS = 32;		///< object knows how to count channels of incoming multi-channel signals

    /**	Header for any non-ui signal processing object.
        For ui objects use #t_pxjbox.
        @ingroup	msp	*/
    struct t_pxobject {
        t_object	z_ob;		///< The standard #t_object struct.
        long		z_in;
        void*		z_proxy;
        long		z_disabled;	///< set to non-zero if this object is muted (using pcontrol or mute~ objects)
        short		z_count;	///< an array that indicates what inlets/outlets are connected with signals
        short		z_misc;		///< flags (bitmask) determining object behaviour, such as #Z_NO_INPLACE.
    };



    // system access prototypes

    BEGIN_USING_C_LINKAGE

    /**	A function pointer for the audio perform routine used by MSP objects to process blocks of samples. @ingroup msp */
    typedef void (*t_perfroutine64)(t_object* x, t_object* dsp64, double** ins, long numins, double** outs, long numouts, long sampleframes, long flags, void* userparam);


    // access to DSP system variables

    /**	Query MSP for the maximum global vector (block) size.
        @ingroup	msp
        @return		The maximum global vector size for the MSP environment.		*/
    int sys_getmaxblksize(void);

    /**	Query MSP for the current global vector (block) size.
        @ingroup	msp
        @return		The current global vector size for the MSP environment.		*/
    int sys_getblksize(void);

    /**	Query MSP for the global sample rate.
        @ingroup	msp
        @return		The global sample rate of the MSP environment.		*/
    float sys_getsr(void);


    /**	Query MSP to determine whether or not it is running.
        @ingroup	msp
        @return		Returns true if the DSP is turned on, otherwise returns false.		*/
    int sys_getdspstate(void);		// returns whether audio is on or off

    /** Query MSP to determine whether or not a given audio object is
        in a running dsp chain.  This is preferable over sys_getdspstate()
        since global audio can be on but an object could be in a patcher that
        is not running.
        @ingroup	msp
        @return		Returns true if the MSP object is in a patcher that has audio on, otherwise returns false.
    */
    int sys_getdspobjdspstate(t_object* o);


    void dsp_add64(t_object* chain, t_object* x, t_perfroutine64 f, long flags, void* userparam);

    /**	Call this routine after creating your object in the new instance routine
        with object_alloc(). Cast your object to #t_pxobject as the first
        argument, then specify the number of signal inputs your object will
        have. dsp_setup() initializes fields of the #t_pxobject header and
        allocates any proxies needed (if num_signal_inputs is greater than 1).

        Some signal objects have no inputs; you should pass 0 for
        num_signal_inputs in this case. After calling dsp_setup(), you can
        create additional non-signal inlets using intin(), floatin(), or
        inlet_new().

        @ingroup			msp
        @param	x			Your object's pointer.
        @param	nsignals	The number of signal/proxy inlets to create for the object.
        @see 				#dsp_setup	*/
    void z_dsp_setup(t_pxobject* x, long nsignals);		// called in new method

        /**	This is commonly used rather than directly calling z_dsp_setup() in MSP objects.
             @ingroup	msp	*/
        inline void dsp_setup(t_pxobject* x, long nsignals) {
            z_dsp_setup(x, nsignals);
        }

    void dsp_resize(t_pxobject* x, long nsignals); // for dynamic inlets


    /**	This function disposes of any memory used by proxies allocated by
        dsp_setup(). It also notifies the signal compiler that the DSP call chain
        needs to be rebuilt if signal processing is active. You should be sure to
        call this before de-allocating any memory that might be in use by your
        object’s perform routine, in the event that signal processing is on when
        your object is freed.

        @ingroup	msp
        @param	x	The object to free.
        @see		#dsp_free	*/
    void z_dsp_free(t_pxobject* x);

        /**	This is commonly used rather than directly calling z_dsp_free() in MSP objects.
             @ingroup	msp	*/
        inline void dsp_free(t_pxobject* x) {
            z_dsp_free(x);
        }

    /**	This routine must be called in your object's initialization routine. It
        adds a set of methods to your object's class that are called by MSP to
        build the DSP call chain. These methods function entirely
        transparently to your object so you don't have to worry about them.
        However, you should avoid binding anything to their names: signal,
        userconnect, nsiginlets, and enable.

        This routine is for non-user-interface objects only
        (where the first item in your object's struct is a t_pxobject).
        It must be called prior to calling class_register() for your class.

        @ingroup	msp
        @param	c	The class to make dsp-ready.
        @see		class_dspinitjbox()	*/
    void class_dspinit(t_class* c);

    /**	This routine must be called in your object's initialization routine. It
        adds a set of methods to your object's class that are called by MSP to
        build the DSP call chain. These methods function entirely
        transparently to your object so you don't have to worry about them.
        However, you should avoid binding anything to their names: signal,
        userconnect, nsiginlets, and enable.

        This routine is for user-interface objects only
        (where the first item in your object's struct is a t_jbox).

        @ingroup	msp
        @param	c	The class to make dsp-ready.
        @see		class_dspinit()	*/
    void class_dspinitjbox(t_class* c);




    END_USING_C_LINKAGE


    //-- ddz, so this is OK, just needs jpatcher_api.h first, right?

    #if defined(_JPATCHER_API_H_) || defined(_DOXY_)
    BEGIN_USING_C_LINKAGE


    /**	Header for any ui signal processing object.
        For non-ui objects use #t_pxobject.
        @ingroup	msp	*/
    struct t_pxjbox {
        t_jbox	z_box;			///< The box struct used by all ui objects.
        long	z_in;
        void*	z_proxy;
        long	z_disabled;		///< set to non-zero if this object is muted (using pcontrol or mute~ objects)
        short	z_count;		///< an array that indicates what inlets/outlets are connected with signals
        short	z_misc;			///< flags (bitmask) determining object behaviour, such as #Z_NO_INPLACE
    };

    void z_jbox_dsp_setup(t_pxjbox* x, long nsignals);
    void z_jbox_dsp_free(t_pxjbox* x);


    inline void dsp_setupjbox(t_pxjbox* x, long nsignals) {
        z_jbox_dsp_setup(x, nsignals);
    }

    inline void dsp_freejbox(t_pxjbox* x) {
        z_jbox_dsp_free(x);
    }


    END_USING_C_LINKAGE


    #endif // _JPATCHER_API_H_



    #ifdef __APPLE
    #pragma mark -
    #pragma mark Buffer API
    #endif


    /**	A buffer~ reference.
        Use this struct to represent a reference to a buffer~ object in Max.
        Use the buffer_ref_getbuffer() call to return a pointer to the buffer.
        You can then make calls on the buffer itself.

        @ingroup buffers
    */
    struct t_buffer_ref;


    /**	A buffer~ object.
        This represents the actual buffer~ object.
        You can use this to send messages, query attributes, etc. of the actual buffer object
        referenced by a #t_buffer_ref.

        @ingroup buffers
     */
    typedef t_object t_buffer_obj;


    /**	Common buffer~ data/metadata.
        This info can be retreived from a buffer~ using the buffer_getinfo() call.

        @ingroup buffers
     */
    struct t_buffer_info {
        t_symbol*	b_name;		///< name of the buffer
        float*		b_samples;		///< stored with interleaved channels if multi-channel
        long		b_frames;		///< number of sample frames (each one is sizeof(float) * b_nchans bytes)
        long		b_nchans;		///< number of channels
        long		b_size;			///< size of buffer in floats
        float		b_sr;			///< sampling rate of the buffer
        long		b_modtime;		///< last modified time ("dirty" method)
        long		b_rfu[57];		///< reserved for future use (total struct size is 64x4 = 256 bytes)
    };


    BEGIN_USING_C_LINKAGE


    /**	Create a reference to a buffer~ object by name.
        You must release the buffer reference using object_free() when you are finished using it.

        @ingroup buffers
        @param	self	pointer to your object
        @param	name 	the name of the buffer~
        @return			a pointer to your new buffer reference
    */
    t_buffer_ref* buffer_ref_new(t_object* self, t_symbol* name);


    /**	Change a buffer reference to refer to a different buffer~ object by name.

         @ingroup buffers
         @param	x		the buffer reference
         @param	name 	the name of a different buffer~ to reference
     */
    void buffer_ref_set(t_buffer_ref* x, t_symbol* name);


    /**	Query to find out if a buffer~ with the referenced name actually exists.

        @ingroup buffers
        @param	x		the buffer reference
        @return			non-zero if the buffer~ exists, otherwise zero
    */
    t_atom_long buffer_ref_exists(t_buffer_ref* x);


    /**	Query a buffer reference to get the actual buffer~ object being referenced, if it exists.

         @ingroup buffers
         @param	x			the buffer reference
         @return			the buffer object if exists, otherwise NULL
     */
    t_buffer_obj* buffer_ref_getobject(t_buffer_ref* x);


    /**	Your object needs to handle notifications issued by the buffer~ you reference.
        You do this by defining a "notify" method.
        Your notify method should then call this notify method for the #t_buffer_ref.

        @ingroup buffers
        @param	x		the buffer reference
        @param	s 		the registered name of the sending object
        @param	msg		then name of the notification/message sent
        @param	sender	the pointer to the sending object
        @param	data	optional argument sent with the notification/message
        @return			a max error code
    */
    t_max_err buffer_ref_notify(t_buffer_ref* x, t_symbol* s, t_symbol* msg, void* sender, void* data);





    /**	Open a viewer window to display the contents of the buffer~.

        @ingroup buffers
        @param	buffer_object	the buffer object
    */
    void buffer_view(t_buffer_obj* buffer_object);


    /**	Claim the buffer~ and get a pointer to the first sample in memory.
        When you are done reading/writing to the buffer you must call buffer_unlocksamples().
        If the attempt to claim the buffer~ fails the returned pointer will be NULL.

        @ingroup buffers
        @param	buffer_object	the buffer object
        @return					a pointer to the first sample in memory, or NULL if the buffer doesn't exist.
    */
    float* buffer_locksamples(t_buffer_obj* buffer_object);


    /**	Release your claim on the buffer~ contents so that other objects may read/write to the buffer~.

        @ingroup buffers
        @param	buffer_object	the buffer object
    */
    void buffer_unlocksamples(t_buffer_obj* buffer_object);


    /**	Query a buffer~ to find out how many channels are present in the buffer content.

        @ingroup buffers
        @param	buffer_object	the buffer object
        @return					the number of channels in the buffer
    */
    t_atom_long buffer_getchannelcount(t_buffer_obj* buffer_object);


    /**	Query a buffer~ to find out how many frames long the buffer content is in samples.

        @ingroup buffers
        @param	buffer_object	the buffer object
        @return					the number of frames in the buffer
    */
    t_atom_long buffer_getframecount(t_buffer_obj* buffer_object);


    /**	Query a buffer~ to find out its native sample rate in samples per second.

        @ingroup buffers
        @param	buffer_object	the buffer object
        @return					the sample rate in samples per second
    */
    t_atom_float buffer_getsamplerate(t_buffer_obj* buffer_object);


    /**	Query a buffer~ to find out its native sample rate in samples per millisecond.

        @ingroup buffers
        @param	buffer_object	the buffer object
        @return					the sample rate in samples per millisecond
    */
    t_atom_float buffer_getmillisamplerate(t_buffer_obj* buffer_object);


    /** Set the number of samples with which to zero-pad the buffer~'s contents.
        The typical application for this need is to pad a buffer with enough room to allow for the reach of a FIR kernel in convolution.

        @ingroup buffers
        @param	buffer_object	the buffer object
        @param	samplecount		the number of sample to pad the buffer with on each side of the contents
        @return					an error code
    */
    t_max_err buffer_setpadding(t_buffer_obj* buffer_object, t_atom_long samplecount);


    /**	Set the buffer's dirty flag, indicating that changes have been made.

        @ingroup buffers
        @param	buffer_object	the buffer object
        @return					an error code
     */
    t_max_err buffer_setdirty(t_buffer_obj* buffer_object);


    /** Retrieve the name of the last file to be read by a buffer~.
        (Not the last file written).

        @ingroup buffers
        @param	buffer_object	the buffer object
        @return					The name of the file last read, or gensym("") if no files have been read.

        @version Introduced in Max 7.0.1
     */
    t_symbol* buffer_getfilename(t_buffer_obj* buffer_object);



    // Internal or low-level functions


    // buffer_perform functions to replace the direct use of
    // atomics and other buffer state flags from the perform method
    // wrapped by buffer_locksamples() and buffer_unlocksamples()
    t_max_err buffer_perform_begin(t_buffer_obj* buffer_object);
    t_max_err buffer_perform_end(t_buffer_obj* buffer_object);

    // utility function for getting buffer info in struct form
    // without needing to know entire buffer struct
    t_max_err buffer_getinfo(t_buffer_obj* buffer_object, t_buffer_info* info);


    // the following functions are not to be called in the perform method
    // please use the lightweight buffer_perform methods

    // use buffer_edit functions to collapse all operations of
    // locking heavy b_mutex, setting b_valid flag,
    // waiting on lightweight atomic b_inuse, etc.
    t_max_err buffer_edit_begin(t_buffer_obj* buffer_object);
    t_max_err buffer_edit_end(t_buffer_obj* buffer_object, long valid);  // valid 0=FALSE, positive=TRUE, negative=RESTORE_OLD_VALID (not common)

    // low level mutex locking used by buffer_edit fucntions.
    // use only if you really know what you're doing.
    // otherwise, use the buffer_edit functions
    // if you're touching a t_buffer outside perform
    t_max_err buffer_lock(t_buffer_obj* buffer_object);
    t_max_err buffer_trylock(t_buffer_obj* buffer_object);
    t_max_err buffer_unlock(t_buffer_obj* buffer_object);

    // low level utilities used by buffer_edit functions
    // use only if you really know what you're doing.
    // otherwise, use the buffer_edit functions
    // if you're touching a t_buffer outside perform
    t_buffer_obj* buffer_findowner(t_buffer_obj* buffer_object);
    long buffer_spinwait(t_buffer_obj* buffer_object);
    long buffer_valid(t_buffer_obj* buffer_object, long way);

    END_USING_C_LINKAGE

}} // namespace c74::max
