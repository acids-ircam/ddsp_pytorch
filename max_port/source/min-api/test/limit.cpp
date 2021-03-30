/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.
#include "catch.hpp"
#include "c74_min_api.h"

using namespace c74::min;


TEST_CASE( "clamping", "[limits]" ) {

    REQUIRE( MIN_CLAMP(54, 0, 25) == 25 );
    REQUIRE( MIN_CLAMP(-54, 0, 25) == 0 );

    REQUIRE( MIN_CLAMP(54.0, 0, 25) == 25 );
    REQUIRE( MIN_CLAMP(-54.0, 0, 25) == 0 );

    SECTION( "side effects" ) {
        auto i = 0;
        REQUIRE( MIN_CLAMP(i++, -4, 4) == 0 );
        REQUIRE( MIN_CLAMP(i++, -4, 4) == 1 );

        // The following does not build with MSVC, claiming eror C2101: '&' on constant
        // REQUIRE( MIN_CLAMP(++i, -4, 4) == 3 );
    }
}


TEST_CASE( "power-of-two", "[limits]" ) {

    REQUIRE( limit_to_power_of_two(0) == 0 );
    REQUIRE( limit_to_power_of_two(1) == 1 );
    REQUIRE( limit_to_power_of_two(2) == 2 );
    REQUIRE( limit_to_power_of_two(3) == 4 );
    REQUIRE( limit_to_power_of_two(4) == 4 );

    REQUIRE( limit_to_power_of_two(54) == 64 );
    REQUIRE( limit_to_power_of_two(64) == 64 );
    REQUIRE( limit_to_power_of_two(65) == 128 );

    REQUIRE( limit_to_power_of_two(-5) == 0 );

    // Type must be integral... thus floats won't compile
    // REQUIRE( limit_to_power_of_two(3.14) == 4 );
}


TEST_CASE( "wrapping", "[limits]" ) {

    SECTION( "common use -- wrapping degrees around a circle" ) {
        REQUIRE( wrap(360, 0, 360) == 0);
        REQUIRE( wrap(720, 0, 360) == 0);
        REQUIRE( wrap(-720, 0, 360) == 0);
        REQUIRE( wrap(-719, 0, 360) == 1);
        REQUIRE( wrap(-10, 0, 360) == 350);
    }

    SECTION( "common use -- wrapping radians around a circle" ) {
        REQUIRE( wrap(0.0, -M_PI, M_PI) == 0);
        REQUIRE( wrap(-M_PI, -M_PI, M_PI) == -M_PI);
        REQUIRE( wrap(M_PI, -M_PI, M_PI) == -M_PI);
        REQUIRE( wrap(-M_PI_2, -M_PI, M_PI) == -M_PI_2);
        REQUIRE( wrap(M_PI_2, -M_PI, M_PI) == M_PI_2);
    }

    SECTION( "one end is positive, the other is zero" ) {
        REQUIRE( wrap(  0.0,  0.0,  10.0) ==  0.0 );
        REQUIRE( wrap( 10.0,  0.0,  10.0) ==  0.0 );
        REQUIRE( wrap(  5.0,  0.0,  10.0) ==  5.0 );
        REQUIRE( wrap( 15.0,  0.0,  10.0) ==  5.0 );
        REQUIRE( wrap( -1.0,  0.0,  10.0) ==  9.0 );
        REQUIRE( wrap(-15.0,  0.0,  10.0) ==  5.0 );
    }

    SECTION( "bottom end is negative, top end of the range is positive" ) {
        REQUIRE( wrap(  0.0, -5.0,   5.0) ==  0.0 );
        REQUIRE( wrap(  5.0, -5.0,   5.0) == -5.0 );
        REQUIRE( wrap(  0.0, -5.0,   5.0) ==  0.0 );
        REQUIRE( wrap( 10.0, -5.0,   5.0) ==  0.0 );
        REQUIRE( wrap( -6.0, -5.0,   5.0) ==  4.0 );
        REQUIRE( wrap(-10.0, -5.0,   5.0) ==  0.0 );
        REQUIRE( wrap( 24.0, -5.0,   5.0) ==  4.0 );
    }

    SECTION( "one end is negative, the other is zero" ) {
        REQUIRE( wrap(  0.0, -5.0,   0.0) == -5.0 );
        REQUIRE( wrap(  5.0, -5.0,   0.0) == -5.0 );
        REQUIRE( wrap( 10.0, -5.0,   0.0) == -5.0 );
        REQUIRE( wrap( -3.0, -5.0,   0.0) == -3.0 );
        REQUIRE( wrap( -6.0, -5.0,   0.0) == -1.0 );
        REQUIRE( wrap(-10.0, -5.0,   0.0) == -5.0 );
    }

    SECTION( "both ends of the range are positive" ) {
        REQUIRE( wrap<number>(  0.0,   5, 25) == 20.0 );
        REQUIRE( wrap<number>(  5.0,   5, 25) == 5.0 );
        REQUIRE( wrap<number>( 10.0,   5, 25) == 10.0 );
        REQUIRE( wrap<number>( 25.0,   5, 25) == 5.0 );
        REQUIRE( wrap<number>( -6.0,   5, 25) == 14.0 );
        REQUIRE( wrap<number>( 1001.0, 5, 25) == Approx(21.0) );
    }

    SECTION( "both ends of the range are negative, and the order of the range is reversed" ) {
        REQUIRE( wrap<number>(  0.0,   -5, -25) == -20.0 );
        REQUIRE( wrap<number>(  5.0,   -5, -25) == -15.0 );
        REQUIRE( wrap<number>( -10.0,   -5, -25) == -10.0 );
        REQUIRE( wrap<number>( -5.0,   -5, -25) == -25.0 );
        REQUIRE( wrap<number>( -36.0,   -5, -25) == -16.0 );
        REQUIRE( wrap<number>( 1001.0, -5, -25) == Approx(-19.0) );
    }
}


TEST_CASE( "folding", "[limits]" ) {
    SECTION( "common use -- frequency mimics aliasing" ) {
        REQUIRE( fold(0.0, 0.0, 22050.0)		== 0.0);
        REQUIRE( fold(-10.0, 0.0, 22050.0)		== 10.0);
        REQUIRE( fold(22060.0, 0.0, 22050.0)	== 22040.0);
        REQUIRE( fold(44100.0, 0.0, 22050.0)	== 0.0);
        REQUIRE( fold(44200.0, 0.0, 22050.0)	== 100.0);
        REQUIRE( fold(-22050.0, 0.0, 22050.0)	== 22050.0);
    }

    SECTION( "both ends of the range are positive" ) {
        REQUIRE( fold<number>(  0.0,   5, 25) == 10.0 );
        REQUIRE( fold<number>(  5.0,   5, 25) == 5.0 );
        REQUIRE( fold<number>( 10.0,   5, 25) == 10.0 );
        REQUIRE( fold<number>( 25.0,   5, 25) == 25.0 );
        REQUIRE( fold<number>( -6.0,   5, 25) == 16.0 );
        REQUIRE( fold<number>( 1000.0, 5, 25) == Approx(10.0) );
    }

    SECTION( "both ends of the range are negative, and the order of the range is reversed" ) {
        REQUIRE( fold<number>(  0.0,   -5, -25) == -10.0 );
        REQUIRE( fold<number>(  5.0,   -5, -25) == -15.0 );
        REQUIRE( fold<number>( -10.0,  -5, -25) == -10.0 );
        REQUIRE( fold<number>( -5.0,   -5, -25) == -5.0 );
        REQUIRE( fold<number>( -36.0,  -5, -25) == -14.0 );
        REQUIRE( fold<number>( 1001.0, -5, -25) == Approx(-11.0) );
    }
}


TEST_CASE( "scaling", "[limits]" ) {
    REQUIRE( scale(0.0, 0.0, 1.0, 0.0, 127.0) == 0.0);
    REQUIRE( scale(1.0, 0.0, 1.0, 0.0, 127.0) == 127.0);
    REQUIRE( scale(0.5, 0.0, 1.0, 0.0, 127.0) == 63.5);
    REQUIRE( scale(2.0, 0.0, 1.0, 0.0, 127.0) == 254.0);
    REQUIRE( scale(-1.0, 0.0, 1.0, 0.0, 127.0) == -127.0);

    REQUIRE(scale(1.0, 1.0, 1.0, 0.0, 127.0) == 0.0);
    REQUIRE(scale(0.0, 0.0, 0.0, 0.0, 127.0) == 0.0);

    // the same but with integers

    REQUIRE( scale(0, 0, 1, 0, 127) == 0);
    REQUIRE( scale(1, 0, 1, 0, 127) == 127);
    REQUIRE( scale(2, 0, 1, 0, 127) == 254);
    REQUIRE( scale(-1, 0, 1, 0, 127) == -127);

    // now with an exponential curve

    REQUIRE( scale(0.0, 0.0, 1.0, 0.0, 127.0, 2.0) == 0.0);
    REQUIRE( scale(1.0, 0.0, 1.0, 0.0, 127.0, 2.0) == 127.0);
    REQUIRE( scale(0.5, 0.0, 1.0, 0.0, 127.0, 2.0) == 31.75);
    REQUIRE( scale(2.0, 0.0, 1.0, 0.0, 127.0, 2.0) == 508.0);
    REQUIRE( scale(-1.0, 0.0, 1.0, 0.0, 127.0, 2.0) == -127.0);

    // and with a log curve

    REQUIRE( scale(0.0, 0.0, 1.0, 0.0, 127.0, 0.5) == 0.0);
    REQUIRE( scale(1.0, 0.0, 1.0, 0.0, 127.0, 0.5) == 127.0);
    REQUIRE( scale(0.5, 0.0, 1.0, 0.0, 127.0, 0.5) == Approx(89.8025612107));
    REQUIRE( scale(2.0, 0.0, 1.0, 0.0, 127.0, 0.5) == Approx(179.6051224214));
    REQUIRE( scale(-1.0, 0.0, 1.0, 0.0, 127.0, 0.5) == -127.0);
}
