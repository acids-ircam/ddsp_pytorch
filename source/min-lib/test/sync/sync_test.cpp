/// @file
///	@brief 		Unit test for the sync class
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#define CATCH_CONFIG_MAIN
#include "c74_min_catch.h"


SCENARIO ("responds appropriately to messages and attrs") {

    GIVEN ("An instance of the sync class") {
        c74::min::lib::sync	s;

        REQUIRE( s.phase() == 0.0 );

        WHEN ("phase is set within a valid range") {
            s.phase(0.25);
            THEN("phase is set to the value specified")
            REQUIRE( s.phase() == Approx(0.25) );
        }
        AND_WHEN ("setting phase param over range") {
            s.phase(1.3);
            THEN("phase is wrapped into range (integral part is discarded)" )
            REQUIRE( s.phase() == Approx(0.3) );
        }
        AND_WHEN ("setting phase param way over range") {
            s.phase(2.45);
            THEN("phase is wrapped into range (integral part is discarded)" )
            REQUIRE( s.phase() == Approx(0.45) );
        }
        AND_WHEN ("setting phase param under range") {
            s.phase(-1.3);
            THEN( "phase is wrapped into range" )
            REQUIRE( s.phase() == Approx(0.7) );
        }

        // There is no gain or offset parameter because we are calculating single-sample and
        // this class is kept to its essential function

        WHEN ("frequency is set within a valid range") {
            s.frequency(1000.0, 96000.0);
            THEN("frequency is set to the value specified")
            REQUIRE( s.frequency() == Approx(1000.0) );
        }
        AND_WHEN ("setting frequency param way above range 1") {
            s.frequency(50000.0, 96000.0);
            THEN("frequency is folded back down into range")
            REQUIRE( s.frequency() == Approx(46000.0) );
        }
        AND_WHEN ("setting frequency param way above range 2") {
            s.frequency(98000.0, 96000.0);
            THEN("frequency is folded back down into range")
            REQUIRE( s.frequency() == Approx(-2000.0) );
        }
        AND_WHEN ("setting frequency param below range") {
            s.frequency(-2000.0, 96000.0);
            THEN("frequency is set to the value specified")
            REQUIRE( s.frequency() == Approx(-2000.0) );
        }
    }
}

TEST_CASE ("Confirm sample operator changes the phase") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an sync instance with frequency set to 1.0");

    c74::min::lib::sync	s;
    s.frequency(1.0, 256.0);
    sample output;

    INFO ("After 64 samples of output, phase should be 0.25");
    for (int i=0; i < 64; i++)
    {
        output = s();
    }
    REQUIRE( s.phase() == Approx(0.25) );	// check the new value for phase

    INFO ("After another 64 samples of output, phase should be 0.5");
    for (int i=0; i < 64; i++)
    {
        output = s();
    }
    REQUIRE( s.phase() == Approx(0.5) );	// check the new value for phase

    INFO ("After another 128 samples of output, phase should be 1.0");
    for (int i=0; i < 128; i++)
    {
        output = s();
    }
    REQUIRE( s.phase() == Approx(1.0) );	// check the new value for phase

    INFO ("After another 64 samples of output, phase should be back at 0.25");
    for (int i=0; i < 64; i++)
    {
        output = s();
    }
    REQUIRE( s.phase() == Approx(0.25) );	// check the new value for phase


}

