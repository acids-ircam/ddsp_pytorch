/// @file
///	@brief 		Unit test for the oscillator filter class
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#define CATCH_CONFIG_MAIN
#include "c74_min_catch.h"


TEST_CASE ("Produce correctly sized wavetables using constructor parameters") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with no arguments, creates a wavetable with 4096 samples");

    oscillator<> o;

    REQUIRE( o.size() == 4096 );		// check the defaults

    INFO ("Using oscillator with argument set to 512, creates a properly sized wavetable");

    oscillator<> o2(512);

    REQUIRE( o2.size() == 512 );		// check the defaults

    INFO ("Using oscillator with argument set to 16384, creates a properly sized wavetable");

    oscillator<> o3(16384);

    REQUIRE( o3.size() == 16384 );		// check the defaults

}


TEST_CASE ("Confirm changes to frequency and phase parameters") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with no arguments, which should set default values of 0.0 phase, 0.0 frequency");

    oscillator<> o;

    REQUIRE( o.phase() == 0.0 );		// check the defaults
    REQUIRE( o.frequency() == 0.0 );	// ..

    INFO ("Changing the phase to 0.25");
    o.phase(0.25);
    REQUIRE( o.phase() == Approx(0.25) );	// check the new value for phase
    REQUIRE( o.frequency() == 0.0 );		// and confirm the default frequency remains

    INFO ("Changing the phase to 1.35 should wrap back around");
    o.phase(1.32);
    REQUIRE( o.phase() == Approx(0.32) );	// check the new value for phase
    REQUIRE( o.frequency() == 0.0 );		// and confirm the default frequency remains

    INFO ("Changing the phase to -0.15 should wrap back around");
    o.phase(-0.17);
    REQUIRE( o.phase() == Approx(0.83) );	// check the new value for phase
    REQUIRE( o.frequency() == 0.0 );		// and confirm the default frequency remains

    INFO ("Changing the frequency to 2000.0 at 96k sampling rate");
    o.frequency(2000.0, 96000.0);
    REQUIRE( o.frequency() == Approx(2000.0) );	// check the new value for frequency
    REQUIRE( o.phase() == Approx(0.83) );		// and confirm the phase remains unchanged

    INFO ("Changing the frequency to 2000.0 at 96k sampling rate");
    o.frequency(2000.0, 96000.0);
    REQUIRE( o.frequency() == Approx(2000.0) );	// check the new value for frequency
    REQUIRE( o.phase() == Approx(0.83) );		// and confirm the phase remains unchanged

    INFO ("Trying to set frequency above the Nyquist, which should fold back below");
    o.frequency(52000.0, 96000.0);
    REQUIRE( o.frequency() == Approx(44000.0) );	// check the new value for frequency
    REQUIRE( o.phase() == Approx(0.83) );		// and confirm the phase remains unchanged

    INFO ("Trying to set frequency above the sampling rate, which should fold back below");
    o.frequency(100000.0, 96000.0);
    REQUIRE( o.frequency() == Approx(-4000.0) );	// check the new value for frequency
    REQUIRE( o.phase() == Approx(0.83) );		// and confirm the phase remains unchanged

    INFO ("Trying to set frequency to negative value, which should be allowed");
    o.frequency(-6000.0, 96000.0);
    REQUIRE( o.frequency() == Approx(-6000.0) );	// check the new value for frequency
    REQUIRE( o.phase() == Approx(0.83) );		// and confirm the phase remains unchanged

}


TEST_CASE ("Confirm sample operator changes the phase") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 256");

    oscillator<> o { 256 };
    o.frequency(1.0, 256.0);
    sample output;

    INFO ("After 64 samples of output, phase should be 0.25");
    for (int i=0; i < 64; i++)
    {
        output = o();
    }
    REQUIRE( o.phase() == Approx(0.25) );	// check the new value for phase

    INFO ("After another 64 samples of output, phase should be 0.5");
    for (int i=0; i < 64; i++)
    {
        output = o();
    }
    REQUIRE( o.phase() == Approx(0.5) );	// check the new value for phase

    INFO ("After another 128 samples of output, phase should be 1.0");
    for (int i=0; i < 128; i++)
    {
        output = o();
    }
    REQUIRE( o.phase() == Approx(1.0) );	// check the new value for phase

    INFO ("After another 64 samples of output, phase should be back at 0.25");
    for (int i=0; i < 64; i++)
    {
        output = o();
    }
    REQUIRE( o.phase() == Approx(0.25) );	// check the new value for phase


}

TEST_CASE ("Produce a sine waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 64, default waveform of sine");

    oscillator<> o { 64 };

    INFO ("Setting frequency to produce single cycle with duration of 64 samples");
    o.frequency(1.0, 64.0);

    // The following output was generated using the Octave code in GeneratorTargetOutput.m by NW
    c74::min::sample_vector reference = {
        0,
        0.0980171403295606,
        0.1950903220161282,
        0.2902846772544623,
        0.3826834323650898,
        0.4713967368259976,
        0.5555702330196022,
        0.6343932841636455,
        0.7071067811865475,
        0.7730104533627369,
        0.8314696123025451,
        0.8819212643483549,
        0.9238795325112867,
        0.9569403357322089,
        0.9807852804032304,
        0.9951847266721968,
        1,
        0.9951847266721969,
        0.9807852804032304,
        0.9569403357322089,
        0.9238795325112867,
        0.881921264348355,
        0.8314696123025455,
        0.7730104533627371,
        0.7071067811865476,
        0.6343932841636455,
        0.5555702330196022,
        0.4713967368259978,
        0.3826834323650898,
        0.2902846772544623,
        0.1950903220161286,
        0.09801714032956084,
        1.224646799147353e-16,
        -0.09801714032956059,
        -0.1950903220161284,
        -0.2902846772544622,
        -0.3826834323650897,
        -0.4713967368259976,
        -0.555570233019602,
        -0.6343932841636453,
        -0.7071067811865475,
        -0.7730104533627367,
        -0.8314696123025452,
        -0.8819212643483549,
        -0.9238795325112865,
        -0.9569403357322088,
        -0.9807852804032303,
        -0.9951847266721969,
        -1,
        -0.9951847266721969,
        -0.9807852804032304,
        -0.9569403357322089,
        -0.9238795325112866,
        -0.881921264348355,
        -0.8314696123025456,
        -0.7730104533627369,
        -0.7071067811865477,
        -0.6343932841636459,
        -0.5555702330196022,
        -0.4713967368259979,
        -0.3826834323650904,
        -0.2902846772544624,
        -0.1950903220161287,
        -0.09801714032956052
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    REQUIRE_VECTOR_APPROX( output, reference );


}


TEST_CASE ("Produce a ramp waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 64, setting waveform to ramp");

    oscillator<> o { 64 };
    o.change_waveform<generator::ramp<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        -1,
        -0.9682539682539683,
        -0.9365079365079365,
        -0.9047619047619048,
        -0.873015873015873,
        -0.8412698412698413,
        -0.8095238095238095,
        -0.7777777777777778,
        -0.746031746031746,
        -0.7142857142857143,
        -0.6825396825396826,
        -0.6507936507936508,
        -0.6190476190476191,
        -0.5873015873015873,
        -0.5555555555555556,
        -0.5238095238095238,
        -0.4920634920634921,
        -0.4603174603174603,
        -0.4285714285714286,
        -0.3968253968253969,
        -0.3650793650793651,
        -0.3333333333333334,
        -0.3015873015873016,
        -0.2698412698412699,
        -0.2380952380952381,
        -0.2063492063492064,
        -0.1746031746031746,
        -0.1428571428571429,
        -0.1111111111111112,
        -0.07936507936507942,
        -0.04761904761904767,
        -0.01587301587301593,
        0.01587301587301582,
        0.04761904761904767,
        0.07936507936507931,
        0.1111111111111112,
        0.1428571428571428,
        0.1746031746031746,
        0.2063492063492063,
        0.2380952380952381,
        0.2698412698412698,
        0.3015873015873016,
        0.3333333333333333,
        0.3650793650793651,
        0.3968253968253967,
        0.4285714285714286,
        0.4603174603174602,
        0.4920634920634921,
        0.5238095238095237,
        0.5555555555555556,
        0.5873015873015872,
        0.6190476190476191,
        0.6507936507936507,
        0.6825396825396826,
        0.7142857142857142,
        0.746031746031746,
        0.7777777777777777,
        0.8095238095238095,
        0.8412698412698412,
        0.873015873015873,
        0.9047619047619047,
        0.9365079365079365,
        0.9682539682539681,
        1
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    REQUIRE_VECTOR_APPROX( output, reference );


}


TEST_CASE ("Produce a sawtooth waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 64, setting waveform to sawtooth");

    oscillator<> o { 64 };
    o.change_waveform<generator::sawtooth<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        -1,
        -0.96875,
        -0.9375,
        -0.90625,
        -0.875,
        -0.84375,
        -0.8125,
        -0.78125,
        -0.75,
        -0.71875,
        -0.6875,
        -0.65625,
        -0.625,
        -0.59375,
        -0.5625,
        -0.53125,
        -0.5,
        -0.46875,
        -0.4375,
        -0.40625,
        -0.375,
        -0.34375,
        -0.3125,
        -0.28125,
        -0.25,
        -0.21875,
        -0.1875,
        -0.15625,
        -0.125,
        -0.09375,
        -0.0625,
        -0.03125,
        0,
        0.03125,
        0.0625,
        0.09375,
        0.125,
        0.15625,
        0.1875,
        0.21875,
        0.25,
        0.28125,
        0.3125,
        0.34375,
        0.375,
        0.40625,
        0.4375,
        0.46875,
        0.5,
        0.53125,
        0.5625,
        0.59375,
        0.625,
        0.65625,
        0.6875,
        0.71875,
        0.75,
        0.78125,
        0.8125,
        0.84375,
        0.875,
        0.90625,
        0.9375,
        0.96875
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    REQUIRE_VECTOR_APPROX( output, reference );


}


TEST_CASE ("Produce a unipolar ramp waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 64, setting waveform to unipolar ramp");

    oscillator<> o { 64 };
    o.change_waveform<generator::ramp_unipolar<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        0,
        0.01587301587301587,
        0.03174603174603174,
        0.04761904761904762,
        0.06349206349206349,
        0.07936507936507936,
        0.09523809523809523,
        0.1111111111111111,
        0.126984126984127,
        0.1428571428571428,
        0.1587301587301587,
        0.1746031746031746,
        0.1904761904761905,
        0.2063492063492063,
        0.2222222222222222,
        0.2380952380952381,
        0.253968253968254,
        0.2698412698412698,
        0.2857142857142857,
        0.3015873015873016,
        0.3174603174603174,
        0.3333333333333333,
        0.3492063492063492,
        0.3650793650793651,
        0.3809523809523809,
        0.3968253968253968,
        0.4126984126984127,
        0.4285714285714285,
        0.4444444444444444,
        0.4603174603174603,
        0.4761904761904762,
        0.492063492063492,
        0.5079365079365079,
        0.5238095238095238,
        0.5396825396825397,
        0.5555555555555556,
        0.5714285714285714,
        0.5873015873015873,
        0.6031746031746031,
        0.6190476190476191,
        0.6349206349206349,
        0.6507936507936508,
        0.6666666666666666,
        0.6825396825396826,
        0.6984126984126984,
        0.7142857142857143,
        0.7301587301587301,
        0.746031746031746,
        0.7619047619047619,
        0.7777777777777778,
        0.7936507936507936,
        0.8095238095238095,
        0.8253968253968254,
        0.8412698412698413,
        0.8571428571428571,
        0.873015873015873,
        0.8888888888888888,
        0.9047619047619048,
        0.9206349206349206,
        0.9365079365079365,
        0.9523809523809523,
        0.9682539682539683,
        0.9841269841269841,
        1
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    REQUIRE_VECTOR_APPROX( output, reference );


}


TEST_CASE ("Produce a unipolar sawtooth waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 64, setting waveform to unipolar sawtooth");

    oscillator<> o { 64 };
    o.change_waveform<generator::sawtooth_unipolar<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        0,
        0.015625,
        0.03125,
        0.046875,
        0.0625,
        0.078125,
        0.09375,
        0.109375,
        0.125,
        0.140625,
        0.15625,
        0.171875,
        0.1875,
        0.203125,
        0.21875,
        0.234375,
        0.25,
        0.265625,
        0.28125,
        0.296875,
        0.3125,
        0.328125,
        0.34375,
        0.359375,
        0.375,
        0.390625,
        0.40625,
        0.421875,
        0.4375,
        0.453125,
        0.46875,
        0.484375,
        0.5,
        0.515625,
        0.53125,
        0.546875,
        0.5625,
        0.578125,
        0.59375,
        0.609375,
        0.625,
        0.640625,
        0.65625,
        0.671875,
        0.6875,
        0.703125,
        0.71875,
        0.734375,
        0.75,
        0.765625,
        0.78125,
        0.796875,
        0.8125,
        0.828125,
        0.84375,
        0.859375,
        0.875,
        0.890625,
        0.90625,
        0.921875,
        0.9375,
        0.953125,
        0.96875,
        0.984375
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    REQUIRE_VECTOR_APPROX( output, reference );


}


TEST_CASE ("Produce a unipolar sine waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 64, setting waveform to unipolar sine");

    oscillator<> o { 64 };
    o.change_waveform<generator::sine_unipolar<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        0.5,
        0.5490085701647803,
        0.5975451610080641,
        0.6451423386272311,
        0.6913417161825449,
        0.7356983684129988,
        0.7777851165098011,
        0.8171966420818227,
        0.8535533905932737,
        0.8865052266813684,
        0.9157348061512726,
        0.9409606321741775,
        0.9619397662556434,
        0.9784701678661045,
        0.9903926402016152,
        0.9975923633360984,
        1,
        0.9975923633360985,
        0.9903926402016152,
        0.9784701678661045,
        0.9619397662556434,
        0.9409606321741775,
        0.9157348061512727,
        0.8865052266813686,
        0.8535533905932737,
        0.8171966420818227,
        0.7777851165098011,
        0.735698368412999,
        0.6913417161825449,
        0.6451423386272311,
        0.5975451610080643,
        0.5490085701647804,
        0.5000000000000001,
        0.4509914298352197,
        0.4024548389919358,
        0.3548576613727689,
        0.3086582838174552,
        0.2643016315870012,
        0.222214883490199,
        0.1828033579181774,
        0.1464466094067263,
        0.1134947733186317,
        0.08426519384872738,
        0.05903936782582253,
        0.03806023374435674,
        0.02152983213389559,
        0.00960735979838484,
        0.002407636663901536,
        0,
        0.002407636663901536,
        0.009607359798384785,
        0.02152983213389553,
        0.03806023374435669,
        0.05903936782582248,
        0.08426519384872722,
        0.1134947733186316,
        0.1464466094067262,
        0.182803357918177,
        0.2222148834901989,
        0.264301631587001,
        0.3086582838174548,
        0.3548576613727688,
        0.4024548389919356,
        0.4509914298352197
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    REQUIRE_VECTOR_APPROX( output, reference );


}


TEST_CASE ("Produce a cosine waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 64, setting waveform to cosine");

    oscillator<> o { 64 };
    o.change_waveform<generator::cosine<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        1,
        0.9951847266721969,
        0.9807852804032304,
        0.9569403357322088,
        0.9238795325112867,
        0.881921264348355,
        0.8314696123025452,
        0.773010453362737,
        0.7071067811865476,
        0.6343932841636456,
        0.5555702330196023,
        0.4713967368259978,
        0.3826834323650898,
        0.2902846772544623,
        0.1950903220161283,
        0.09801714032956077,
        6.123233995736766e-17,
        -0.09801714032956066,
        -0.1950903220161282,
        -0.2902846772544622,
        -0.3826834323650897,
        -0.4713967368259977,
        -0.555570233019602,
        -0.6343932841636453,
        -0.7071067811865475,
        -0.773010453362737,
        -0.8314696123025453,
        -0.8819212643483549,
        -0.9238795325112867,
        -0.9569403357322088,
        -0.9807852804032304,
        -0.9951847266721968,
        -1,
        -0.9951847266721969,
        -0.9807852804032304,
        -0.9569403357322089,
        -0.9238795325112867,
        -0.881921264348355,
        -0.8314696123025455,
        -0.7730104533627371,
        -0.7071067811865477,
        -0.6343932841636459,
        -0.5555702330196022,
        -0.4713967368259979,
        -0.3826834323650903,
        -0.2902846772544624,
        -0.1950903220161287,
        -0.09801714032956045,
        -1.83697019872103e-16,
        0.09801714032956009,
        0.1950903220161283,
        0.2902846772544621,
        0.38268343236509,
        0.4713967368259976,
        0.5555702330196018,
        0.6343932841636456,
        0.7071067811865475,
        0.7730104533627367,
        0.8314696123025452,
        0.8819212643483549,
        0.9238795325112865,
        0.9569403357322088,
        0.9807852804032303,
        0.9951847266721969
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    REQUIRE_VECTOR_APPROX( output, reference );


}


TEST_CASE ("Produce a unipolar cosine waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 64, setting waveform to unipolar cosine");

    oscillator<> o { 64 };
    o.change_waveform<generator::cosine_unipolar<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        1,
        0.9975923633360985,
        0.9903926402016152,
        0.9784701678661044,
        0.9619397662556434,
        0.9409606321741775,
        0.9157348061512727,
        0.8865052266813684,
        0.8535533905932737,
        0.8171966420818229,
        0.7777851165098011,
        0.735698368412999,
        0.6913417161825449,
        0.6451423386272311,
        0.5975451610080642,
        0.5490085701647804,
        0.5,
        0.4509914298352197,
        0.4024548389919359,
        0.3548576613727689,
        0.3086582838174551,
        0.2643016315870012,
        0.222214883490199,
        0.1828033579181774,
        0.1464466094067263,
        0.1134947733186315,
        0.08426519384872733,
        0.05903936782582253,
        0.03806023374435663,
        0.02152983213389559,
        0.009607359798384785,
        0.002407636663901591,
        0,
        0.002407636663901536,
        0.009607359798384785,
        0.02152983213389553,
        0.03806023374435663,
        0.05903936782582248,
        0.08426519384872727,
        0.1134947733186314,
        0.1464466094067262,
        0.182803357918177,
        0.2222148834901989,
        0.264301631587001,
        0.3086582838174549,
        0.3548576613727688,
        0.4024548389919357,
        0.4509914298352198,
        0.4999999999999999,
        0.54900857016478,
        0.5975451610080641,
        0.645142338627231,
        0.691341716182545,
        0.7356983684129987,
        0.7777851165098009,
        0.8171966420818229,
        0.8535533905932737,
        0.8865052266813683,
        0.9157348061512727,
        0.9409606321741775,
        0.9619397662556433,
        0.9784701678661044,
        0.9903926402016152,
        0.9975923633360985
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    REQUIRE_VECTOR_APPROX( output, reference );


}


TEST_CASE ("Produce a triangle waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 64, setting waveform to triangle");

    oscillator<> o { 64 };
    o.change_waveform<generator::triangle<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        0,
        0.0625,
        0.125,
        0.1875,
        0.25,
        0.3125,
        0.375,
        0.4375,
        0.5,
        0.5625,
        0.625,
        0.6875,
        0.75,
        0.8125,
        0.875,
        0.9375,
        1,
        0.9375,
        0.875,
        0.8125,
        0.75,
        0.6875,
        0.625,
        0.5625,
        0.5,
        0.4375,
        0.375,
        0.3125,
        0.25,
        0.1875,
        0.125,
        0.0625,
        0,
        -0.0625,
        -0.125,
        -0.1875,
        -0.25,
        -0.3125,
        -0.375,
        -0.4375,
        -0.5,
        -0.5625,
        -0.625,
        -0.6875,
        -0.75,
        -0.8125,
        -0.875,
        -0.9375,
        -1,
        -0.9375,
        -0.875,
        -0.8125,
        -0.75,
        -0.6875,
        -0.625,
        -0.5625,
        -0.5,
        -0.4375,
        -0.375,
        -0.3125,
        -0.25,
        -0.1875,
        -0.125,
        -0.0625
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    REQUIRE_VECTOR_APPROX( output, reference );


}


TEST_CASE ("Produce a unipolar triangle waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 64, setting waveform to unipolar triangle");

    oscillator<> o { 64 };
    o.change_waveform<generator::triangle_unipolar<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        0.5,
        0.53125,
        0.5625,
        0.59375,
        0.625,
        0.65625,
        0.6875,
        0.71875,
        0.75,
        0.78125,
        0.8125,
        0.84375,
        0.875,
        0.90625,
        0.9375,
        0.96875,
        1,
        0.96875,
        0.9375,
        0.90625,
        0.875,
        0.84375,
        0.8125,
        0.78125,
        0.75,
        0.71875,
        0.6875,
        0.65625,
        0.625,
        0.59375,
        0.5625,
        0.53125,
        0.5,
        0.46875,
        0.4375,
        0.40625,
        0.375,
        0.34375,
        0.3125,
        0.28125,
        0.25,
        0.21875,
        0.1875,
        0.15625,
        0.125,
        0.09375,
        0.0625,
        0.03125,
        0,
        0.03125,
        0.0625,
        0.09375,
        0.125,
        0.15625,
        0.1875,
        0.21875,
        0.25,
        0.28125,
        0.3125,
        0.34375,
        0.375,
        0.40625,
        0.4375,
        0.46875
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    REQUIRE_VECTOR_APPROX( output, reference );


}

TEST_CASE ("Interpolate values between samples in unipolar sawtooth waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 32, setting waveform to unipolar sawtooth");

    oscillator<> o { 32 };
    o.change_waveform<generator::sawtooth_unipolar<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples, forcing oscillator to interpolate");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        0,
        -0.109375, // different from 64-item wavetable
        0.03125,
        0.046875,
        0.0625,
        0.078125,
        0.09375,
        0.109375,
        0.125,
        0.140625,
        0.15625,
        0.171875,
        0.1875,
        0.203125,
        0.21875,
        0.234375,
        0.25,
        0.265625,
        0.28125,
        0.296875,
        0.3125,
        0.328125,
        0.34375,
        0.359375,
        0.375,
        0.390625,
        0.40625,
        0.421875,
        0.4375,
        0.453125,
        0.46875,
        0.484375,
        0.5,
        0.515625,
        0.53125,
        0.546875,
        0.5625,
        0.578125,
        0.59375,
        0.609375,
        0.625,
        0.640625,
        0.65625,
        0.671875,
        0.6875,
        0.703125,
        0.71875,
        0.734375,
        0.75,
        0.765625,
        0.78125,
        0.796875,
        0.8125,
        0.828125,
        0.84375,
        0.859375,
        0.875,
        0.890625,
        0.90625,
        0.921875,
        0.9375,
        1.078125, // different from 64-item wavetable
        0.96875,
        0.484375 // different from 64-item wavetable
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        //std::cout << y << ", ";
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    // reference output was copied from earlier 64-item wavetable test
    // 3 edge cases caused by discontinuous shape have been noted
    REQUIRE_VECTOR_APPROX( output, reference );


}

TEST_CASE ("Interpolate values between samples in cosine waveform") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an oscillator instance with wavetable size of 32, setting waveform to cosine");

    oscillator<> o { 32 };
    o.change_waveform<generator::cosine<>>();

    INFO ("Setting frequency to produce single cycle with duration of 64 samples, forcing oscillator to interpolate");
    o.frequency(1.0, 64.0);

    c74::min::sample_vector reference = {
        1,
        0.999908,
        0.980785,
        0.961482,
        0.92388,
        0.886107,
        0.83147,
        0.776679,
        0.707107,
        0.637404,
        0.55557,
        0.473634,
        0.382683,
        0.291662,
        0.19509,
        0.0984823,
        6.12323e-17,
        -0.0984823,
        -0.19509,
        -0.291662,
        -0.382683,
        -0.473634,
        -0.55557,
        -0.637404,
        -0.707107,
        -0.776679,
        -0.83147,
        -0.886107,
        -0.92388,
        -0.961482,
        -0.980785,
        -0.999908,
        -1,
        -0.999908,
        -0.980785,
        -0.961482,
        -0.92388,
        -0.886107,
        -0.83147,
        -0.776679,
        -0.707107,
        -0.637404,
        -0.55557,
        -0.473634,
        -0.382683,
        -0.291662,
        -0.19509,
        -0.0984823,
        -1.83697e-16,
        0.0984823,
        0.19509,
        0.291662,
        0.382683,
        0.473634,
        0.55557,
        0.637404,
        0.707107,
        0.776679,
        0.83147,
        0.886107,
        0.92388,
        0.961482,
        0.980785,
        0.999908
    };

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : reference) {
        auto y = o();
        output.push_back(y);
        //std::cout << y << ", ";
        x = 0; // unused
    }

    INFO("The output matches an externally generated reference set.");
    // reference output was captured using std::cout
    // line is now commented out within the above for loop
    REQUIRE_VECTOR_APPROX( output, reference );


}
