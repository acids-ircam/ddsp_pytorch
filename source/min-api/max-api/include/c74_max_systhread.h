/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {


    BEGIN_USING_C_LINKAGE


    /** An opaque thread instance pointer.
        @ingroup threading
    */
    typedef void *t_systhread;


    /** An opaque mutex handle.
        @ingroup threading
    */
    typedef void *t_systhread_mutex;


    /** An opaque cond handle.
        @ingroup threading
    */
    typedef void *t_systhread_cond;

    typedef void *t_systhread_rwlock;

    typedef void *t_systhread_key;

    /** systhread_mutex_new() flags
        @ingroup threading
    */
    typedef enum {
        SYSTHREAD_MUTEX_NORMAL =		0x00000000,	///< Normal
        SYSTHREAD_MUTEX_ERRORCHECK =	0x00000001,	///< Error-checking
        SYSTHREAD_MUTEX_RECURSIVE =		0x00000002	///< Recursive
    } e_max_systhread_mutex_flags;

    typedef enum {
        SYSTHREAD_PRIORITY_MIN = -30,
        SYSTHREAD_PRIORITY_DEFAULT = 0,
        SYSTHREAD_PRIORITY_MAX = 30
    } e_max_systhread_priority;

    typedef enum {
        SYSTHREAD_RWLOCK_NORMAL =		0x00000000,
        SYSTHREAD_RWLOCK_LITE =			0x00000001
    } e_max_systhread_rwlock_flags;

    /**
         Create a new thread.
        @ingroup threading

        @param	entryproc	A method to call in the new thread when the thread is created.
        @param	arg			An argument to pass to the method specified for entryproc.
                            Typically this might be a pointer to your object's struct.
        @param	stacksize	Not used.  Pass 0 for this argument.
        @param	priority	Pass 0 for default priority.  The priority can range from -32 to 32 where -32 is low, 0 is default and 32 is high.
        @param	flags		Not used.  Pass 0 for this argument.
        @param	thread		The address of a #t_systhread where this thread's instance pointer will be stored.
        @return				A Max error code as defined in #e_max_errorcodes.
    */
    long systhread_create(method entryproc, void *arg, long stacksize, long priority, long flags, t_systhread *thread);


    /**
         Forcefully kill a thread -- not recommended.
        @ingroup threading

        @param	thread	The thread to kill.
        @return			A Max error code as defined in #e_max_errorcodes.
    */
    long systhread_terminate(t_systhread thread);


    /**
         Suspend the execution of the calling thread.
        @ingroup threading

        @param	milliseconds	The number of milliseconds to suspend the execution of the calling thread.
                                The actual amount of time may be longer depending on various factors.
    */
    void systhread_sleep(long milliseconds);


    /**
         Exit the calling thread.
        Call this from within a thread made using systhread_create() when the thread is no longer needed.

        @ingroup threading
        @param	status		You will typically pass 0 for status.
                            This value will be accessible by systhread_join(), if needed.
    */
    void systhread_exit(long status);


    /**
        Wait for thread to quit and get return value from systhread_exit().

        @ingroup threading
        @param	thread		The thread to join.
        @param	retval		The address of a long to hold the return value (status) from systhread_exit().
        @return				A Max error code as defined in #e_max_errorcodes.

        @remark	If your object is freed, and your thread function accesses memory from your object,
                then you will obviously have a memory violation.
                A common use of systhread_join() is to prevent this situation by waiting (in your free method)
                for the thread to exit.
    */
    long systhread_join(t_systhread thread, unsigned int* retval);

    /**
        Detach a thread. After detaching a thread you cannot call systhread_join() on it.

        @ingroup threading
        @param	thread		The thread to join.
        @return				A Max error code as defined in #e_max_errorcodes.

         @remark	You should either call systhread_join() on a thread or systhread_detach()
                    to allow the system to reclaim resources.
     */
    long systhread_detach(t_systhread thread);

    /**
        Return the thread instance pointer for the calling thread.
        @ingroup	threading
        @return		The thread instance pointer for the thread from which this function is called.
    */
    t_systhread systhread_self(void);

    /**
        Set the thread priority for the given thread.
        @ingroup	threading
        @param		thread 			The thread for which to set the priority.
        @param		priority		A value in the range -32 to 32 where -32 is lowest, 0 is default, and 32 is highest.
    */
    void systhread_setpriority(t_systhread thread, int priority);

    /**
        Get the thread priority for the given thread.
        @ingroup	threading
        @param		thread 			The thread for which to find the priority.
        @return						The current priority value for the given thread.
    */
    int systhread_getpriority(t_systhread thread);

    char *systhread_getstackbase(void);


    // private
    void systhread_init(void);
    void systhread_mainstacksetup(void);
    void systhread_timerstacksetup(void);
    short systhread_stackcheck(void);


    /** Check to see if the function currently being executed is in the main thread.
        @ingroup	threading
        @return		Returns true if the function is being executed in the main thread, otherwise false.
    */
    short systhread_ismainthread(void);


    /** Check to see if the function currently being executed is in a scheduler thread.
        @ingroup	threading
        @return		Returns true if the function is being executed in a scheduler thread, otherwise false.
    */
    short systhread_istimerthread(void);


    /** Check to see if the function currently being executed is in an audio thread.
     @ingroup	threading
     @return		Returns true if the function is being executed in an audio thread, otherwise false.
     */

    short systhread_isaudiothread(void);


    /**
        Create a new mutex, which can be used to place thread locks around critical code.
        The mutex should be freed with systhread_mutex_free().
        @ingroup mutex

        @param	pmutex	The address of a variable to store the mutex pointer.
        @param	flags	Flags to determine the behaviour of the mutex, as defined in #e_max_systhread_mutex_flags.
        @return			A Max error code as defined in #e_max_errorcodes.

        @remark			One reason to use systhread_mutex_new() instead of @ref critical is to
                        create non-recursive locks, which are lighter-weight than recursive locks.
    */
    long systhread_mutex_new(t_systhread_mutex *pmutex,long flags);


    /**
        Free a mutex created with systhread_mutex_new().
        @ingroup mutex
        @param	pmutex	The mutex instance pointer.
        @return			A Max error code as defined in #e_max_errorcodes.
    */
    long systhread_mutex_free(t_systhread_mutex pmutex);


    /**
        Enter block of locked code code until a systhread_mutex_unlock() is reached.
        It is important to keep the code in this block as small as possible.
        @ingroup mutex
        @param	pmutex	The mutex instance pointer.
        @return			A Max error code as defined in #e_max_errorcodes.
        @see			systhread_mutex_trylock()
    */
    long systhread_mutex_lock(t_systhread_mutex pmutex);


    /**
        Exit a block of code locked with systhread_mutex_lock().
        @ingroup mutex
        @param	pmutex	The mutex instance pointer.
        @return			A Max error code as defined in #e_max_errorcodes.
    */
    long systhread_mutex_unlock(t_systhread_mutex pmutex);


    /**
        Try to enter block of locked code code until a systhread_mutex_unlock() is reached.
        If the lock cannot be entered, this function will return non-zero.

        @ingroup mutex
        @param	pmutex	The mutex instance pointer.
        @return			Returns non-zero if there was a problem entering.
        @see			systhread_mutex_lock()
    */
    long systhread_mutex_trylock(t_systhread_mutex pmutex);


    /**
        Convenience utility that combines systhread_mutex_new() and systhread_mutex_lock().
        @ingroup mutex
        @param	pmutex	The address of a variable to store the mutex pointer.
        @param	flags	Flags to determine the behaviour of the mutex, as defined in #e_max_systhread_mutex_flags.
        @return			A Max error code as defined in #e_max_errorcodes.
    */
    long systhread_mutex_newlock(t_systhread_mutex *pmutex,long flags);

    t_max_err systhread_rwlock_new(t_systhread_rwlock *rwlock, long flags);
    t_max_err systhread_rwlock_free(t_systhread_rwlock rwlock);
    t_max_err systhread_rwlock_rdlock(t_systhread_rwlock rwlock);
    t_max_err systhread_rwlock_tryrdlock(t_systhread_rwlock rwlock);
    t_max_err systhread_rwlock_rdunlock(t_systhread_rwlock rwlock);
    t_max_err systhread_rwlock_wrlock(t_systhread_rwlock rwlock);
    t_max_err systhread_rwlock_trywrlock(t_systhread_rwlock rwlock);
    t_max_err systhread_rwlock_wrunlock(t_systhread_rwlock rwlock);
    t_max_err systhread_rwlock_setspintime(t_systhread_rwlock rwlock, double spintime_ms);
    t_max_err systhread_rwlock_getspintime(t_systhread_rwlock rwlock, double *spintime_ms);

    long systhread_cond_new(t_systhread_cond *pcond, long flags);
    long systhread_cond_free(t_systhread_cond pcond);
    long systhread_cond_wait(t_systhread_cond pcond, t_systhread_mutex pmutex);
    long systhread_cond_signal(t_systhread_cond pcond);
    long systhread_cond_broadcast(t_systhread_cond pcond);

    long systhread_key_create(t_systhread_key *key, void (*destructor)(void*));
    long systhread_key_delete(t_systhread_key key);
    void* systhread_getspecific(t_systhread_key key);
    long systhread_setspecific(t_systhread_key key, const void *value);

    END_USING_C_LINKAGE

}} // namespace c74::max

