/// @file
///	@brief 		Unit test for the allpass filter class
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#define CATCH_CONFIG_MAIN
#include "c74_min_catch.h"


TEST_CASE ("Produce the correct impulse response using default constructor") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an allpass instance with no arguments, which sets default gain at 0.0 and capacity to 4410 samples");

    allpass	f;
    REQUIRE( f.gain() == 0.0 );		// check the defaults
    REQUIRE( f.delay() == 4410 );	// ...

    INFO("Changing the delay time to 1 sample & gain to 0.5...");
    f.delay(1);
    f.gain(0.5);
    REQUIRE( f.delay() == 1 );     // check the new value
    REQUIRE( f.gain() == 0.5 );		// check the new value

    INFO ("And then pushing a 64-sample impulse through the unit...");
    sample_vector zero(64, 0.0);
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : impulse) {
        auto y = f(x);
        output.push_back(y);
    }

    INFO ("And finally checking our output against a reference produced with Octave.");
    // coefficients calculated in Octave using:
    //  pkg load signal
    //	a = [-0.5, 1.0] % fir coefficients
    //  b = [1.0, -0.5] % iir coefficients
    //  reference = impz(a,b,64)

    sample_vector reference = {
        -0.5,
        0.75,
        0.375,
        0.1875,
        0.09375,
        0.046875,
        0.0234375,
        0.01171875,
        0.005859375,
        0.0029296875,
        0.00146484375,
        0.000732421875,
        0.0003662109375,
        0.00018310546875,
        9.1552734375e-05,
        4.57763671875e-05,
        2.288818359375e-05,
        1.1444091796875e-05,
        5.7220458984375e-06,
        2.86102294921875e-06,
        1.430511474609375e-06,
        7.152557373046875e-07,
        3.576278686523438e-07,
        1.788139343261719e-07,
        8.940696716308594e-08,
        4.470348358154297e-08,
        2.235174179077148e-08,
        1.117587089538574e-08,
        5.587935447692871e-09,
        2.793967723846436e-09,
        1.396983861923218e-09,
        6.984919309616089e-10,
        3.492459654808044e-10,
        1.746229827404022e-10,
        8.731149137020111e-11,
        4.365574568510056e-11,
        2.182787284255028e-11,
        1.091393642127514e-11,
        5.456968210637569e-12,
        2.728484105318785e-12,
        1.364242052659392e-12,
        6.821210263296962e-13,
        3.410605131648481e-13,
        1.70530256582424e-13,
        8.526512829121202e-14,
        4.263256414560601e-14,
        2.131628207280301e-14,
        1.06581410364015e-14,
        5.329070518200751e-15,
        2.664535259100376e-15,
        1.332267629550188e-15,
        6.661338147750939e-16,
        3.33066907387547e-16,
        1.665334536937735e-16,
        8.326672684688674e-17,
        4.163336342344337e-17,
        2.081668171172169e-17,
        1.040834085586084e-17,
        5.204170427930421e-18,
        2.602085213965211e-18,
        1.301042606982605e-18,
        6.505213034913027e-19,
        3.252606517456513e-19,
        1.626303258728257e-19
    };

    // check it
    REQUIRE_VECTOR_APPROX(output, reference);

}

TEST_CASE ("Produce the correct impulse response after constructor with initial values") {

        using namespace c74::min;
        using namespace c74::min::lib;
        INFO ("Using an allpass instance with arguments, which should set default gain at 0.0 and new capacity to 10 samples");

        allpass	f { 10 };
        REQUIRE( f.gain() == 0.0 );		// check the default value
        REQUIRE( f.delay() == 10 );     // check the initialized value

        INFO("Changing the delay time to 1 sample & gain to 0.5...");
        f.delay(1);
        f.gain(0.5);
        REQUIRE( f.delay() == 1 );     // check the new value
        REQUIRE( f.gain() == 0.5 );		// check the new value

        INFO ("And then pushing a 64-sample impulse through the unit...");
        sample_vector impulse(64, 0.0);
        impulse[0] = 1.0;

        // output from our object's processing
        sample_vector	output;

        // run the calculations
        for (auto x : impulse) {
            auto y = f(x);
            output.push_back(y);
        }

        INFO ("And finally checking our output against a reference produced with Octave.");
        // coefficients calculated in Octave using:
        //  pkg load signal
        //	a = [-0.5, 1.0] % fir coefficients
        //  b = [1.0, -0.5] % iir coefficients
        //  reference = impz(a,b,64)

        sample_vector reference = {
            -0.5,
            0.75,
            0.375,
            0.1875,
            0.09375,
            0.046875,
            0.0234375,
            0.01171875,
            0.005859375,
            0.0029296875,
            0.00146484375,
            0.000732421875,
            0.0003662109375,
            0.00018310546875,
            9.1552734375e-05,
            4.57763671875e-05,
            2.288818359375e-05,
            1.1444091796875e-05,
            5.7220458984375e-06,
            2.86102294921875e-06,
            1.430511474609375e-06,
            7.152557373046875e-07,
            3.576278686523438e-07,
            1.788139343261719e-07,
            8.940696716308594e-08,
            4.470348358154297e-08,
            2.235174179077148e-08,
            1.117587089538574e-08,
            5.587935447692871e-09,
            2.793967723846436e-09,
            1.396983861923218e-09,
            6.984919309616089e-10,
            3.492459654808044e-10,
            1.746229827404022e-10,
            8.731149137020111e-11,
            4.365574568510056e-11,
            2.182787284255028e-11,
            1.091393642127514e-11,
            5.456968210637569e-12,
            2.728484105318785e-12,
            1.364242052659392e-12,
            6.821210263296962e-13,
            3.410605131648481e-13,
            1.70530256582424e-13,
            8.526512829121202e-14,
            4.263256414560601e-14,
            2.131628207280301e-14,
            1.06581410364015e-14,
            5.329070518200751e-15,
            2.664535259100376e-15,
            1.332267629550188e-15,
            6.661338147750939e-16,
            3.33066907387547e-16,
            1.665334536937735e-16,
            8.326672684688674e-17,
            4.163336342344337e-17,
            2.081668171172169e-17,
            1.040834085586084e-17,
            5.204170427930421e-18,
            2.602085213965211e-18,
            1.301042606982605e-18,
            6.505213034913027e-19,
            3.252606517456513e-19,
            1.626303258728257e-19
        };


        // check it
        REQUIRE_VECTOR_APPROX(output, reference);

}


TEST_CASE ("Using other non-default settings") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an allpass instance with arguments");

    allpass	f { 4800 };

    INFO("Confirming the size has been set correctly to match the initial argument.");
    REQUIRE( f.delay() == 4800 );

    INFO("Decreasing the delay time to 1000 samples, which is less than capacity...");
    f.delay(1000);
    REQUIRE( f.delay() == 1000 );     // check the new value

    INFO("Increasing the delay time to 2000 samples, which is higher but still less than capacity...");
    f.delay(2000);
    REQUIRE( f.delay() == 2000 );     // check the new value

    INFO("Decreasing the delay time to 2 samples, which is very small...");
    f.delay(2);
    REQUIRE( f.delay() == 2 );     // check the new value


}

TEST_CASE ("Produce the correct impulse response for 2 samples delay, 0.75 gain") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an allpass instance with arguments, which should set default gain at 0.0 and new capacity to 2 samples");

    allpass	f { 2 };
    REQUIRE( f.gain() == 0.0 );		// check the default value
    REQUIRE( f.delay() == 2 );     // check the initialized value

    INFO("Changing the gain to 0.75...");
    f.gain(0.75);
    REQUIRE( f.gain() == 0.75 );		// check the new value

    INFO ("And then pushing a 64-sample impulse through the unit...");
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : impulse) {
        auto y = f(x);
        output.push_back(y);
    }

    INFO ("And finally checking our output against a reference produced with Octave.");
    // coefficients calculated in Octave using:
    //  pkg load signal
    //	a = [-0.75, 0.0, 1.0] % fir coefficients
    //  b = [1.0, 0.0, -0.75] % iir coefficients
    //  reference = impz(a,b,64)

    sample_vector reference = {
        -0.75,
        0,
        0.4375,
        0,
        0.328125,
        0,
        0.24609375,
        0,
        0.1845703125,
        0,
        0.138427734375,
        0,
        0.10382080078125,
        0,
        0.0778656005859375,
        0,
        0.05839920043945312,
        0,
        0.04379940032958984,
        0,
        0.03284955024719238,
        0,
        0.02463716268539429,
        0,
        0.01847787201404572,
        0,
        0.01385840401053429,
        0,
        0.01039380300790071,
        0,
        0.007795352255925536,
        0,
        0.005846514191944152,
        0,
        0.004384885643958114,
        0,
        0.003288664232968586,
        0,
        0.002466498174726439,
        0,
        0.001849873631044829,
        0,
        0.001387405223283622,
        0,
        0.001040553917462717,
        0,
        0.0007804154380970374,
        0,
        0.000585311578572778,
        0,
        0.0004389836839295835,
        0,
        0.0003292377629471877,
        0,
        0.0002469283222103907,
        0,
        0.0001851962416577931,
        0,
        0.0001388971812433448,
        0,
        0.0001041728859325086,
        0,
        7.812966444938144e-05,
        0
    };

    // check it
    REQUIRE_VECTOR_APPROX(output, reference);
}

TEST_CASE ("Produce the correct impulse response for for 4 samples delay, 0.5 gain") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an allpass instance with arguments, which should set default gain at 0.0 and new capacity to 4 samples");

    allpass	f { 4 };
    REQUIRE( f.gain() == 0.0 );		// check the default value
    REQUIRE( f.delay() == 4 );     // check the initialized value

    INFO("Changing the gain to 0.5...");
    f.gain(0.5);
    REQUIRE( f.gain() == 0.5 );		// check the new value

    INFO ("And then pushing a 64-sample impulse through the unit...");
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : impulse) {
        auto y = f(x);
        output.push_back(y);
    }

    INFO ("And finally checking our output against a reference produced with Octave.");
    // coefficients calculated in Octave using:
    //  pkg load signal
    //	a = [-0.5, 0.0, 0.0, 0.0, 1.0] % fir coefficients
    //  b = [1.0, 0.0, 0.0, 0.0, -0.5] % iir coefficients
    //  reference = impz(a,b,64)

    sample_vector reference = {
        -0.5,
        0,
        0,
        0,
        0.75,
        0,
        0,
        0,
        0.375,
        0,
        0,
        0,
        0.1875,
        0,
        0,
        0,
        0.09375,
        0,
        0,
        0,
        0.046875,
        0,
        0,
        0,
        0.0234375,
        0,
        0,
        0,
        0.01171875,
        0,
        0,
        0,
        0.005859375,
        0,
        0,
        0,
        0.0029296875,
        0,
        0,
        0,
        0.00146484375,
        0,
        0,
        0,
        0.000732421875,
        0,
        0,
        0,
        0.0003662109375,
        0,
        0,
        0,
        0.00018310546875,
        0,
        0,
        0,
        9.1552734375e-05,
        0,
        0,
        0,
        4.57763671875e-05,
        0,
        0,
        0
    };

    // check it
    REQUIRE_VECTOR_APPROX(output, reference);
}

TEST_CASE ("Produce the correct impulse response for for 6 samples delay, -0.25 gain") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an allpass instance with arguments, which should set default gain at 0.0 and new capacity to 6 samples");

    allpass	f { 6 };
    REQUIRE( f.gain() == 0.0 );		// check the default value
    REQUIRE( f.delay() == 6 );     // check the initialized value

    INFO("Changing the gain to -0.25...");
    f.gain(-0.25);
    REQUIRE( f.gain() == -0.25 );		// check the new value

    INFO ("And then pushing a 64-sample impulse through the unit...");
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : impulse) {
        auto y = f(x);
        output.push_back(y);
    }

    INFO ("And finally checking our output against a reference produced with Octave.");
    // coefficients calculated in Octave using:
    //  pkg load signal
    //	a = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] % fir coefficients
    //  b = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25] % iir coefficients
    //  reference = impz(a,b,64)

    sample_vector reference = {
        0.25,
        0,
        0,
        0,
        0,
        0,
        0.9375,
        0,
        0,
        0,
        0,
        0,
        -0.234375,
        0,
        0,
        0,
        0,
        0,
        0.05859375,
        0,
        0,
        0,
        0,
        0,
        -0.0146484375,
        0,
        0,
        0,
        0,
        0,
        0.003662109375,
        0,
        0,
        0,
        0,
        0,
        -0.00091552734375,
        0,
        0,
        0,
        0,
        0,
        0.0002288818359375,
        0,
        0,
        0,
        0,
        0,
        -5.7220458984375e-05,
        0,
        0,
        0,
        0,
        0,
        1.430511474609375e-05,
        0,
        0,
        0,
        0,
        0,
        -3.576278686523438e-06,
        0,
        0,
        0
    };

    // check it
    REQUIRE_VECTOR_APPROX(output, reference);
}

TEST_CASE ("Produce the correct impulse response for 7 samples delay, -0.65 gain") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an allpass instance with arguments, which should set default gain at 0.0 and new capacity to 7 samples");

    allpass	f { 7 };
    REQUIRE( f.gain() == 0.0 );		// check the default value
    REQUIRE( f.delay() == 7 );     // check the initialized value

    INFO("Changing the gain to -0.65...");
    f.gain(-0.65);
    REQUIRE( f.gain() == -0.65 );		// check the new value

    INFO ("And then pushing a 64-sample impulse through the unit...");
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : impulse) {
        auto y = f(x);
        output.push_back(y);
    }

    INFO ("And finally checking our output against a reference produced with Octave.");
    // coefficients calculated in Octave using:
    //  pkg load signal
    //	a = [0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] % fir coefficients
    //  b = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.65] % iir coefficients
    //  reference = impz(a,b,64)

    sample_vector reference = {
        0.65,
        0,
        0,
        0,
        0,
        0,
        0,
        0.5774999999999999,
        0,
        0,
        0,
        0,
        0,
        0,
        -0.375375,
        0,
        0,
        0,
        0,
        0,
        0,
        0.24399375,
        0,
        0,
        0,
        0,
        0,
        0,
        -0.1585959375,
        0,
        0,
        0,
        0,
        0,
        0,
        0.103087359375,
        0,
        0,
        0,
        0,
        0,
        0,
        -0.06700678359375001,
        0,
        0,
        0,
        0,
        0,
        0,
        0.04355440933593751,
        0,
        0,
        0,
        0,
        0,
        0,
        -0.02831036606835938,
        0,
        0,
        0,
        0,
        0,
        0,
        0.0184017379444336
    };

    // check it
    REQUIRE_VECTOR_APPROX(output, reference);
}

TEST_CASE ("Produce the correct impulse response for for 8 samples delay, 0.05 gain") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an allpass instance with arguments, which should set default gain at 0.0 and new capacity to 8 samples");

    allpass	f { 8 };
    REQUIRE( f.gain() == 0.0 );		// check the default value
    REQUIRE( f.delay() == 8 );     // check the initialized value

    INFO("Changing the gain to 0.05...");
    f.gain(0.05);
    REQUIRE( f.gain() == 0.05 );		// check the new value

    INFO ("And then pushing a 64-sample impulse through the unit...");
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : impulse) {
        auto y = f(x);
        output.push_back(y);
    }

    INFO ("And finally checking our output against a reference produced with Octave.");
    // coefficients calculated in Octave using:
    //  pkg load signal
    //	a = [-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] % fir coefficients
    //  b = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05] % iir coefficients
    //  reference = impz(a,b,64)

    sample_vector reference = {
        -0.05,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.9975000000000001,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.049875,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.00249375,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.0001246875,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        6.234375000000001e-06,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3.117187500000001e-07,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1.55859375e-08,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    };

    // check it
    REQUIRE_VECTOR_APPROX(output, reference);
}

TEST_CASE ("Produce the correct impulse response for for 12 samples delay, 0.99 gain") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an allpass instance with arguments, which should set default gain at 0.0 and new capacity to 12 samples");

    allpass	f { 12 };
    REQUIRE( f.gain() == 0.0 );		// check the default value
    REQUIRE( f.delay() == 12 );     // check the initialized value

    INFO("Changing the gain to 0.99...");
    f.gain(0.99);
    REQUIRE( f.gain() == 0.99 );		// check the new value

    INFO ("And then pushing a 64-sample impulse through the unit...");
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : impulse) {
        auto y = f(x);
        output.push_back(y);
    }

    INFO ("And finally checking our output against a reference produced with Octave.");
    // coefficients calculated in Octave using:
    //  pkg load signal
    //	a = [-0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] % fir coefficients
    //  b = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.99] % iir coefficients
    //  reference = impz(a,b,64)

    sample_vector reference = {
        -0.99,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01990000000000003,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01970100000000003,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01950399000000003,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01930895010000003,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01911586059900003,
        0,
        0,
        0
    };

    // check it
    REQUIRE_VECTOR_APPROX(output, reference);
}

// NW: developed in response to issue here: https://github.com/nwolek/min-verbs/issues/5
TEST_CASE( "survives a sudden drop in delay time without crashing" ) {

    using namespace c74::min;
    using namespace c74::min::lib;

    number test_size = 44100;

    INFO( "Start with a new instance of the allpass object." );
    // create an instance of our object
    allpass my_object { test_size };

    // create a vector to save output from our object's processing
    sample_vector	output;

    INFO( "Setup an impulse that is just shorter than the allpass circular_storage." );
    // create an impulse buffer to process
    const int		buffersize = 64;
    sample_vector	impulse(buffersize, 0.0);
    impulse[0] = 1.0;



    // change the delay_time to something larger than the default
    INFO( "Change the delay_time attribute to 44 samples (approx 1 ms)." );
    my_object.delay(44);
    my_object.gain(0.5);

    REQUIRE( my_object.delay() == 44 );
    // note that this is our object's attribute only
    // value does not actually get changed for the allpass circular_storage until the next call to its sample_operator

    // run the calculations
    INFO( "Push the impulse sample_vector through the allpass object one time." );
    for (auto i=0; i<10000; ++i)
    for (auto x : impulse) {
        auto y = my_object(x);
        output.push_back(y);
    }


    // change the delay_time to something larger than the default
    INFO( "Change the delay_time attribute UP to 441 samples (approx 10 ms)." );
    my_object.delay(441);
    REQUIRE( my_object.delay() == 441 );

    // run the calculations
    INFO( "Push the impulse sample_vector through the allpass object one time." );
    for (auto i=0; i<10000; ++i)
    for (auto x : impulse) {
        auto y = my_object(x);
        output.push_back(y);
    }


    // change the delay_time to something smaller
    INFO( "Change the delay_time attribute BACK DOWN to 44 samples (approx 1 ms)." );
    my_object.delay(44);
    REQUIRE( my_object.delay() == 44 );

    // run the calculations again, which should wrap around the internal circular_storage
    INFO( "Push the impulse sample_vector through the allpass object another time." );
    for (auto i=0; i<10000; ++i)
    for (auto x : impulse) {
        // CRASH HAPPENS HERE
        auto y = my_object(x);
        output.push_back(y);
    }

    REQUIRE( output != impulse );

}

// NW: developed in response to issue here: https://github.com/Cycling74/min-lib/issues/19
TEST_CASE( "produces valid output after sudden drop in delay time" ) {

    using namespace c74::min;
    using namespace c74::min::lib;

    number test_size = 44100;

    INFO( "Start with a new instance of the allpass object." );
    // create an instance of our object
    allpass my_object { test_size };

    // create a vector to save output from our object's processing
    sample_vector	output;

    INFO( "Setup an impulse that is just shorter than the allpass circular_storage." );
    // create an impulse buffer to process
    const int		buffersize = 64;
    sample_vector	impulse(buffersize, 0.0);
    impulse[0] = 1.0;

    // change the delay_time to something larger than the default
    INFO( "Change the delay_time attribute to 4 samples (approx 0.1 ms)." );
    my_object.delay(4);
    my_object.gain(0.5);

    REQUIRE( my_object.delay() == 4 );
    // note that this is our object's attribute only
    // value does not actually get changed for the allpass circular_storage until the next call to its sample_operator

    // run the calculations
    INFO( "Push the impulse sample_vector through the allpass object one time." );
    for (auto x : impulse) {
        auto y = my_object(x);
        output.push_back(y);
    }


    // change the delay_time to something larger than the default
    INFO( "Change the delay_time attribute UP to 44 samples (approx 1 ms)." );
    my_object.delay(44);
    REQUIRE( my_object.delay() == 44 );

    // run the calculations
    INFO( "Push the impulse sample_vector through the allpass object one time." );
    for (auto x : impulse) {
        auto y = my_object(x);
        output.push_back(y);
    }


    // change the delay_time to something smaller
    INFO( "Change the delay_time attribute BACK DOWN to 4 samples (approx 0.1 ms)." );
    my_object.delay(4);
    REQUIRE( my_object.delay() == 4 );

    // run the calculations again, which should wrap around the internal circular_storage
    INFO( "Push the impulse sample_vector through the allpass object another time." );
    for (auto x : impulse) {
        // CRASH HAPPENS HERE
        auto y = my_object(x);
        output.push_back(y);
    }

#if 0
    INFO( "Compare the last 64 samples to the first 64 samples. They should be the same." );
    for (auto x : impulse) {
        // NW: does not pass, but might no longer be a valid test since memory is no longer zero'ed
        //REQUIRE( output[x] == output[x+128] );
    }
#endif

}

// NW: developed in response to issue here: https://github.com/Cycling74/min-lib/issues/14
TEST_CASE ("Parameters are set properly with different constructors") {

    using namespace c74::min;
    using namespace c74::min::lib;

    INFO ("Creating allpass instance with no arguments, which leaves default capacity = 4410 samples, gain = 0.0");

    allpass	f1;
    REQUIRE( f1.gain() == 0.0 );	// check the defaults
    REQUIRE( f1.delay() == 4410 );	// ...

    INFO ("Creating allpass instance with 1 argument, which sets capacity & size = 200 samples, default gain = 0.0");

    allpass	f2 { 200 };
    REQUIRE( f2.gain() == 0.0 );	// check the default
    REQUIRE( f2.delay() == 200 );	// check the initialized value

    INFO ("Creating allpass instance with 2 arguments, which sets capacity & size = 20 samples, gain = 0.76");

    allpass	f3 { 20, 0.76 };
    REQUIRE( f3.gain() == 0.76 );	// check the initialized value
    REQUIRE( f3.delay() == 20 );	// check the initialized value


    INFO ("Creating allpass instance with 3 arguments, which sets capacity = 400 samples, size = 100, gain = 0.22");

    allpass	f4 { { 400, 100 } , 0.22 };
    REQUIRE( f4.gain() == 0.22 );	// check the initialized value
    REQUIRE( f4.delay() == 100 );	// check the initialized value

}


TEST_CASE ("Setting delay time in milliseconds") {
    using namespace c74::min;
    using namespace c74::min::lib;
    allpass my_allpass;

    number sampling_rate = 44100.0;
    number test_time_1 = 500.0;
    number test_time_1_ms = math::milliseconds_to_samples(test_time_1, sampling_rate);
    number test_time_2 = 1250.0;
    number test_time_2_ms = math::milliseconds_to_samples(test_time_2, sampling_rate);
    number test_time_3 = math::random(400.0,4000.0);
    number test_time_3_ms = math::milliseconds_to_samples(test_time_3, sampling_rate);

    my_allpass.delay_ms(test_time_1, sampling_rate);
    REQUIRE( my_allpass.delay() == test_time_1_ms);

    my_allpass.delay_ms(test_time_2, sampling_rate);
    REQUIRE( my_allpass.delay() == test_time_2_ms);

    my_allpass.delay_ms(test_time_3, sampling_rate);
    REQUIRE( my_allpass.delay() == test_time_3_ms);

}


TEST_CASE ("Produce the correct impulse response for 2.25 samples delay, 0.75 gain") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an allpass instance with arguments, which should set default gain at 0.0 and new capacity to 10 samples");

    allpass	f { 10 };
    REQUIRE( f.gain() == 0.0 );		// check the default value
    REQUIRE( f.delay() == 10 );     // check the initialized value

    INFO("Changing the gain to 0.75...");
    f.gain(0.75);
    REQUIRE( f.gain() == 0.75 );		// check the new value

    INFO("Changing the delay to 2.25 samples...");
    f.delay(2.25);
    REQUIRE( f.delay() == 2.25 );		// check the new value

    INFO ("And then pushing a 64-sample impulse through the unit...");
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : impulse) {
        auto y = f(x);
        output.push_back(y);
        //std::cout << y << ", ";
    }

    INFO ("And finally checking our output against a reference produced earlier.");
    // reference impulse response was captured using std::cout
    // line is now commented out within the above for loop

    sample_vector reference = {
        -0.75,
        -0.0615234,
        0.396137,
        0.047007,
        0.225443,
        0.0979874,
        0.136794,
        0.0995687,
        0.0947645,
        0.0835273,
        0.0718507,
        0.0658151,
        0.0563189,
        0.051084,
        0.0443596,
        0.0396699,
        0.0348411,
        0.0309046,
        0.0272865,
        0.0241284,
        0.021338,
        0.0188556,
        0.0166775,
        0.0147387,
        0.0130337,
        0.0115208,
        0.0101864,
        0.00900508,
        0.00796139,
        0.00703847,
        0.00622255,
        0.00550126,
        0.00486352,
        0.00429976,
        0.00380132,
        0.00336068,
        0.0029711,
        0.00262669,
        0.0023222,
        0.00205301,
        0.00181503,
        0.00160463,
        0.00141862,
        0.00125417,
        0.00110879,
        0.000980258,
        0.000866626,
        0.000766166,
        0.000677352,
        0.000598833,
        0.000529416,
        0.000468046,
        0.00041379,
        0.000365823,
        0.000323417,
        0.000285926,
        0.000252782,
        0.000223479,
        0.000197573,
        0.000174671,
        0.000154423,
        0.000136522,
        0.000120696,
        0.000106705
    };

    // check it
    REQUIRE_VECTOR_APPROX(output, reference);
}


TEST_CASE ("Produce the correct impulse response for 0.0 samples delay, 0.6 gain") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using an allpass instance with no arguments, which should set default gain at 0.0 and capacity at 4410 samples");

    allpass	f { };
    REQUIRE( f.gain() == 0.0 );		// check the default value
    REQUIRE( f.delay() == 4410 );     // check the initialized value

    INFO("Changing the gain to 0.6...");
    f.gain(0.6);
    REQUIRE( f.gain() == 0.6 );		// check the new value

    INFO("Changing the delay to 0.0 samples...");
    f.delay(0.0);
    REQUIRE( f.delay() == 0.0 );		// check the new value

    INFO ("And then pushing a 64-sample impulse through the unit...");
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    // output from our object's processing
    sample_vector	output;

    // run the calculations
    for (auto x : impulse) {
        auto y = f(x);
        output.push_back(y);
        //std::cout << y << ", ";
    }

    INFO ("first test to see if the expected values are in the right place");
    REQUIRE( output[0] == -0.6 );

    INFO ("then test to see if the expected number of non-zeroes were produced");
    int nonzero_count {};
    for (auto& s : output) {
        if (s != 0.0)
            ++nonzero_count;
    }
    REQUIRE( nonzero_count == 1 );

}
