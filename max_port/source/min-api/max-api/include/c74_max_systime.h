/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

    BEGIN_USING_C_LINKAGE

    /**	The Systime data structure.
        @ingroup systime
    */
    struct t_datetime {
        t_uint32	year;			///< year
        t_uint32	month;			///< month
        t_uint32	day;			///< day
        t_uint32	hour;			///< hour
        t_uint32	minute;			///< minute
        t_uint32	second;			///< second
        t_uint32	millisecond;	///< (reserved for future use)
    };


    /**	Flags for the sysdateformat_formatdatetime() function.
        @ingroup systime
    */
    enum e_max_dateflags {
        SYSDATEFORMAT_FLAGS_SHORT = 1,	///< short
        SYSDATEFORMAT_FLAGS_MEDIUM = 2,	///< medium
        SYSDATEFORMAT_FLAGS_LONG = 3	///< long
    };


    /**	Find out the operating system’s time in ticks.
        @ingroup systime
        @return	the system time in ticks.
    */
    t_uint32 systime_ticks(void);


    /**	Find out the operating system’s time in milliseconds.
        Note that this is approximately the number of milliseconds since the OS was started up.
        @ingroup systime
        @return	the system time in milliseconds.
    */
    t_uint32 systime_ms(void);


    /** Find out the current date/time as number of ms since 1/1/1970.
        @ingroup systime
        @return	the number of milliseconds since 1/1/1970.
    */
    t_int64 systime_datetime_milliseconds(void);


    /**	Find out the operating system’s date and time.
        @ingroup systime
        @param	d	Returns the system’s date and time in a #t_datetime data structure.
    */
    void systime_datetime(t_datetime* d);


    /**	Find out the operating system’s time in seconds.
        @ingroup systime
        @return	the system time in seconds.
    */
    t_ptr_uint systime_seconds(void);


    /**	Convert a time in seconds into a #t_datetime representation.
        @ingroup systime
        @param secs	A number of seconds to be represented as a #t_datetime.
        @param d	The address of a #t_datetime that will be filled with the converted value.
    */
    void systime_secondstodate(t_ptr_uint secs, t_datetime* d);


    /**	Convert a #t_datetime representation of time into seconds.
        @ingroup	systime
        @param d	The address of a #t_datetime that contains a valid period of time.
        @return 	The number of seconds represented by d.
    */
    t_ptr_uint systime_datetoseconds(t_datetime* d);


    /**	Fill a #t_datetime struct with a datetime formatted string.
        For example, the string "2007-12-24 12:21:00".
        @ingroup	systime
        @param		strf	A string containing the datetime.
        @param		d		The address of a #t_datetime to fill.
    */
    void sysdateformat_strftimetodatetime(char* strf, t_datetime* d);


    /**	Get a human friendly string representation of a #t_datetime.
        For example: "Today", "Yesterday", etc.
        @ingroup	systime
        @param		d			The address of a #t_datetime to fill.
        @param		dateflags	One of the values defined in #e_max_dateflags.
        @param		timeflags	Currently unused.  Pass 0.
        @param		s			An already allocated string to hold the human friendly result.
        @param		buflen		The number of characters allocated to the string s.
    */
    void sysdateformat_formatdatetime(t_datetime* d, long dateflags, long timeflags, char* s, long buflen);


    END_USING_C_LINKAGE

}} // namespace c74::max
