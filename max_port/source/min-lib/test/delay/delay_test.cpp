/// @file
///	@brief 		Unit test for the delay unit generator
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#define CATCH_CONFIG_MAIN
#include "c74_min_catch.h"


TEST_CASE ("Delay times greater than 1 vector-size") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay instance with no arguments, the default capacity and size are both 256 samples")

    delay my_delay;
    REQUIRE(my_delay.size() == 256);

    INFO ("Changing the delay time to 100 samples...");
    my_delay.size(100.0);

    INFO ("And then pushing an impulse through the unit, using a vector size of 64...");
    sample_vector zero(64, 0.0);
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    INFO ("We process 3 vectors of audio...");
    sample_vector out_samples1;
    sample_vector out_samples2;
    sample_vector out_samples3;

    for (auto& s : impulse)
        out_samples1.push_back( my_delay(s) );
    for (auto& s : zero)
        out_samples2.push_back( my_delay(s) );
    for (auto& s : zero)
        out_samples3.push_back( my_delay(s) );

    INFO ("And check the output is delayed by 100 samples:");
    INFO ("...first 64 samples should all be zero");
    REQUIRE_VECTOR_APPROX( out_samples1 , zero );

    INFO ("...second 64 samples should all be zero except for 100 samples later, which should be 1.0");
    INFO ("   note: this is not the 100th sample, but 100 samples after sample 0 -- a delay of 0 is sample 0");

    for (auto i=0; i<out_samples2.size(); ++i) {
        if (i == 100-64)
            REQUIRE( out_samples2[i] == 1.0 );
        else
            REQUIRE( out_samples2[i] == 0.0 );
    }

    INFO ("...last 64 samples should all be zero");
    REQUIRE_VECTOR_APPROX( out_samples3 , zero );
}


//	Vector Size = 4
//	Delay Size = 6
//	Thus Buffer Size = 10
//
//	Write:	[1][0][0][0][ ][ ][ ][ ][ ][ ]
//	Read:	[ ][ ][ ][ ][x][x][x][x][ ][ ]
//
//	Write:	[ ][ ][ ][ ][0][0][0][0][ ][ ]
//	Read:	[x][x][ ][ ][ ][ ][ ][ ][x][x]
//
//	Write:	[0][0][ ][ ][ ][ ][ ][ ][0][0]
//	Read:	[ ][ ][x][x][x][x][ ][ ][ ][ ]

TEST_CASE ("Delay times greater than 1 vector-size, part 2") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 6 samples, vector-size of 4 samples");

    sample_vector zero(4, 0.0);
    sample_vector impulse(4, 0.0);
    impulse[0] = 1.0;

    delay my_delay(6.0);

    INFO ("We process 3 vectors of audio...");
    sample_vector output[3];
    for (auto& s : impulse)
        output[0].push_back( my_delay(s) );
    for (auto& s : zero)
        output[1].push_back( my_delay(s) );
    for (auto& s : zero)
        output[2].push_back( my_delay(s) );

    REQUIRE( output[0][0] == 0.0 ); // delay 0 samples
    REQUIRE( output[0][1] == 0.0 ); // delay 1 sample
    REQUIRE( output[0][2] == 0.0 ); // delay 2 samples
    REQUIRE( output[0][3] == 0.0 ); // ...

    REQUIRE( output[1][0] == 0.0 );
    REQUIRE( output[1][1] == 0.0 );
    REQUIRE( output[1][2] == 1.0 ); // delay 6 samples
    REQUIRE( output[1][3] == 0.0 );

    REQUIRE( output[2][0] == 0.0 );
    REQUIRE( output[2][1] == 0.0 );
    REQUIRE( output[2][2] == 0.0 );
    REQUIRE( output[2][3] == 0.0 );
}


//	Vector Size = 4
//	Delay Size = 3
//	Thus Buffer Size = 7
//
//	Write:	[1][0][0][0][ ][ ][ ]
//	Read:	[x][ ][ ][ ][x][x][x]
//
//	Write:	[0][ ][ ][ ][0][0][0]
//	Read:	[ ][x][x][x][x][ ][ ]
//
//	Write:	[ ][0][0][0][0][ ][ ]
//	Read:	[x][x][ ][ ][ ][x][x]

TEST_CASE ("Delay times less than 1 vector-size") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 3 samples, vector-size of 4 samples");

    sample_vector zero(4, 0.0);
    sample_vector impulse(4, 0.0);
    impulse[0] = 1.0;

    delay my_delay(3);

    INFO ("We process 3 vectors of audio...");
    sample_vector output[3];
    for (auto& s : impulse)
        output[0].push_back( my_delay(s) );
    for (auto& s : zero)
        output[1].push_back( my_delay(s) );
    for (auto& s : zero)
        output[2].push_back( my_delay(s) );

    REQUIRE( output[0][0] == 0.0 ); // delay 0 samples
    REQUIRE( output[0][1] == 0.0 ); // delay 1 sample
    REQUIRE( output[0][2] == 0.0 ); // delay 2 samples
    REQUIRE( output[0][3] == 1.0 ); // ...

    REQUIRE( output[1][0] == 0.0 );
    REQUIRE( output[1][1] == 0.0 );
    REQUIRE( output[1][2] == 0.0 ); // delay 6 samples
    REQUIRE( output[1][3] == 0.0 );

    REQUIRE( output[2][0] == 0.0 );
    REQUIRE( output[2][1] == 0.0 );
    REQUIRE( output[2][2] == 0.0 );
    REQUIRE( output[2][3] == 0.0 );
}


TEST_CASE ("Delay times less than 1 vector-size, part 2") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 3 samples, vector-size of 16 samples");

    sample_vector	input		{ 0,1,0,0,   0,0,0,0,   2,0,0,0,   0,0,3,0 };
    sample_vector	output;
    delay			my_delay(3);

    INFO ("We process 1 vector of audio...");
    for (auto& s : input)
        output.push_back( my_delay(s) );

    REQUIRE( output[0] == 0.0 ); // delay 0 samples
    REQUIRE( output[1] == 0.0 ); // delay 1 sample
    REQUIRE( output[2] == 0.0 ); // delay 2 samples
    REQUIRE( output[3] == 0.0 ); // ...

    REQUIRE( output[4] == 1.0 );
    REQUIRE( output[5] == 0.0 );
    REQUIRE( output[6] == 0.0 ); // delay 6 samples
    REQUIRE( output[7] == 0.0 );

    REQUIRE( output[8] == 0.0 );
    REQUIRE( output[9] == 0.0 );
    REQUIRE( output[10] == 0.0 );
    REQUIRE( output[11] == 2.0 );

    REQUIRE( output[12] == 0.0 );
    REQUIRE( output[13] == 0.0 );
    REQUIRE( output[14] == 0.0 );
    REQUIRE( output[15] == 0.0 );
}


#ifdef MAC_VERSION
#pragma mark -
#endif


TEST_CASE ("Setting an interpolating delay") {
    using namespace c74::min;
    using namespace c74::min::lib;
    delay my_delay;

    my_delay.size(3.2);
    REQUIRE( my_delay.size() == 3.2 );
    REQUIRE( my_delay.integral_size() == 3 );
    REQUIRE( my_delay.fractional_size() == Approx(0.2) );
}


TEST_CASE ("Setting delay time in milliseconds") {
    using namespace c74::min;
    using namespace c74::min::lib;
    delay my_delay;

    number sampling_rate = 44100.0;
    number test_time_1 = 500.0;
    number test_time_1_ms = math::milliseconds_to_samples(test_time_1, sampling_rate);
    number test_time_2 = 1250.0;
    number test_time_2_ms = math::milliseconds_to_samples(test_time_2, sampling_rate);
    number test_time_3 = math::random(400.0,4000.0);
    number test_time_3_ms = math::milliseconds_to_samples(test_time_3, sampling_rate);

    my_delay.size_ms(test_time_1, sampling_rate);
    REQUIRE( my_delay.size() == test_time_1_ms);

    my_delay.size_ms(test_time_2, sampling_rate);
    REQUIRE( my_delay.size() == test_time_2_ms);

    my_delay.size_ms(test_time_3, sampling_rate);
    REQUIRE( my_delay.size() == test_time_3_ms);

}


TEST_CASE ("Linear interpolation of floating-point delay times") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 2.1 samples, vector-size of 16 samples");

    sample_vector					input		{ 0,1,0,0,		0,0,0,0,	2,0,0,0,		0,0,3,0 };
    sample_vector					output;
    sample_vector					reference	{ 0,0,0,0.9,	0.1,0,0,0,	0,0,1.8,0.2,	0,0,0,0	};
    delay	my_delay(2.1);
    my_delay.change_interpolation(interpolator::type::linear);

    INFO ("We process 1 vector of audio...");
    for (auto& s : input)
        output.push_back( my_delay(s) );

    REQUIRE_VECTOR_APPROX( output , reference );
}


TEST_CASE ("Cubic interpolation of floating-point delay times") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 2.1 samples, vector-size of 16 samples");

    sample_vector	input		{ 0,1,0,0,		0,0,0,0,	2,0,0,0,		0,0,3,0 };
    sample_vector	output;
    sample_vector	reference	{
        0,
        0,
        -0.081000000000000058,
        0.98099999999999998,
        0.1090000000000001,
        -0.0090000000000000149,
        0,
        0,
        0,
        -0.16200000000000012,
        1.962,
        0.21800000000000019,
        -0.01800000000000003,
        0,
        0,
        -0.24300000000000016
    };

    delay	my_delay(2.1);
    my_delay.change_interpolation(interpolator::type::cubic); // default type, so technically not necessary

    INFO ("We process 1 vector of audio...");
    for (auto& s : input)
        output.push_back( my_delay(s) );

    REQUIRE_VECTOR_APPROX( output , reference );
}


TEST_CASE ("Linear interpolation of delay times less than 1 sample") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 0.4 samples, vector-size of 16 samples");

    sample_vector					input		{ 0,1,0,0,		0,0,0,0,	2,0,0,0,		0,0,3,0 };
    sample_vector					output;
    sample_vector					reference	{ 0,0.6,0.4,0,	0,0,0,0,	1.2,0.8,0,0,	0,0,1.8,1.2 };
    delay	my_delay(0.4);
    my_delay.change_interpolation(interpolator::type::linear);

    INFO ("We process 1 vector of audio...");
    for (auto& s : input)
        output.push_back( my_delay(s) );

    REQUIRE_VECTOR_APPROX( output , reference );
}


TEST_CASE ("Cubic interpolation of delay times less than 1 sample") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 0.4 samples, vector-size of 16 samples");

    sample_vector		input		{ 0,1,0,0,	0,0,0,0,	2,0,0,0,	0,0,3,0 };
    sample_vector		output;
    sample_vector		reference	{ // reference is our goal
        0,
        0.74399999999999999,
        0.49600000000000005,
        -0.096000000000000016,
        0,
        0,
        0,
        0,
        1.488,
        0.9920000000000001,
        -0.19200000000000003,
        0,
        0,
        0,
        2.2319999999999998,
        1.4880000000000002,
    };
    delay	my_delay(0.4);
    my_delay.change_interpolation(interpolator::type::cubic);  // default type, so technically not necessary

    INFO ("We process 1 vector of audio...");
    for (auto& s : input)
        output.push_back( my_delay(s) );

    REQUIRE_VECTOR_APPROX( output , reference );
}


TEST_CASE ("Linear interpolation of delay time set to zero") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 0 samples, vector-size of 16 samples");

    sample_vector					input		{ 0,1,0,0,		0,0,0,0,	2,0,0,0,		0,0,3,0 };
    sample_vector					output;
    delay	my_delay(0);
    my_delay.change_interpolation(interpolator::type::linear);

    INFO ("We process 1 vector of audio...");
    for (auto& s : input)
        output.push_back( my_delay(s) );

    REQUIRE_VECTOR_APPROX( output , input );
}


TEST_CASE ("Cubic interpolation of delay time set to zero") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 0 samples, vector-size of 16 samples");

    sample_vector					input		{ 0,1,0,0,		0,0,0,0,	2,0,0,0,		0,0,3,0 };
    sample_vector					output;
    delay	my_delay(0);
    my_delay.change_interpolation(interpolator::type::cubic);  // default type, so technically not necessary

    INFO ("We process 1 vector of audio...");
    for (auto& s : input)
        output.push_back( my_delay(s) );

    REQUIRE_VECTOR_APPROX( output , input );
}


TEST_CASE ("Linear interpolation of delay times greater than 1 vector-size") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 100.2+ samples, vector-size of 64 samples");

    sample_vector zero(64, 0.0);
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    delay my_delay;
    my_delay.change_interpolation(interpolator::type::linear);
    my_delay.size(100.20000000000000284);

    INFO ("We process 3 vectors of audio...");
    sample_vector output[3];
    for (auto& s : impulse)
        output[0].push_back( my_delay(s) );
    for (auto& s : zero)
        output[1].push_back( my_delay(s) );
    for (auto& s : zero)
        output[2].push_back( my_delay(s) );

    INFO ("first test to see if the expected values are in the right place");
    REQUIRE( output[1][36] == 0.79999999999999716 );
    REQUIRE( output[1][37] == 0.20000000000000284 );

    INFO ("then test to see if the expected number of non-zeroes were produced");
    int nonzero_count {};
    for (auto& s : output[0]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    for (auto& s : output[1]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    for (auto& s : output[2]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    REQUIRE( nonzero_count == 2 );
}


TEST_CASE ("Cubic interpolation of delay times greater than 1 vector-size") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 100.2+ samples, vector-size of 64 samples");

    sample_vector zero(64, 0.0);
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    delay my_delay;
    my_delay.change_interpolation(interpolator::type::cubic);	  // default type, so technically not necessary
    my_delay.size(100.20000000000000284);

    INFO ("We process 3 vectors of audio...");
    sample_vector output[3];
    for (auto& s : impulse)
        output[0].push_back( my_delay(s) );
    for (auto& s : zero)
        output[1].push_back( my_delay(s) );
    for (auto& s : zero)
        output[2].push_back( my_delay(s) );

    INFO ("first test to see if the expected values are in the right place");
    REQUIRE( output[1][35] == -0.12800000000000089 );
    REQUIRE( output[1][36] == 0.92799999999999805 );
    REQUIRE( output[1][37] == 0.23200000000000365 );
    REQUIRE( output[1][38] == -0.032000000000000799 );

    INFO ("then test to see if the expected number of non-zeroes were produced");
    int nonzero_count {};
    for (auto& s : output[0]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    for (auto& s : output[1]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    for (auto& s : output[2]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    REQUIRE( nonzero_count == 4 );
}


TEST_CASE ("Linear interpolation of delay times at the edge of a vector") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 63.7+ samples, vector-size of 64 samples");

    sample_vector zero(64, 0.0);
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    delay my_delay;
    my_delay.change_interpolation(interpolator::type::linear);
    my_delay.size(63.70000000000000284);

    INFO ("We process 3 vectors of audio...");
    sample_vector output[3];
    for (auto& s : impulse)
        output[0].push_back( my_delay(s) );
    for (auto& s : zero)
        output[1].push_back( my_delay(s) );
    for (auto& s : zero)
        output[2].push_back( my_delay(s) );

    INFO ("first test to see if the expected values are in the right place");
    REQUIRE( output[0][63] == 0.29999999999999716 );
    REQUIRE( output[1][0] == 0.70000000000000284 );

    INFO ("then test to see if the expected number of non-zeroes were produced");
    int nonzero_count {};
    for (auto& s : output[0]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    for (auto& s : output[1]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    for (auto& s : output[2]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    REQUIRE( nonzero_count == 2 );
}


TEST_CASE ("Cubic interpolation of delay times at the edge of a vector") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 63.7+ samples, vector-size of 64 samples");

    sample_vector zero(64, 0.0);
    sample_vector impulse(64, 0.0);
    impulse[0] = 1.0;

    delay my_delay;
    my_delay.change_interpolation(interpolator::type::cubic);	  // default type, so technically not necessary
    my_delay.size(63.70000000000000284);

    INFO ("We process 3 vectors of audio...");
    sample_vector output[3];
    for (auto& s : impulse)
        output[0].push_back( my_delay(s) );
    for (auto& s : zero)
        output[1].push_back( my_delay(s) );
    for (auto& s : zero)
        output[2].push_back( my_delay(s) );

    INFO ("first test to see if the expected values are in the right place");
    REQUIRE( output[0][62] == -0.062999999999999057 );
    REQUIRE( output[0][63] == 0.36299999999999621 );
    REQUIRE( output[1][0] == 0.84700000000000263 );
    REQUIRE( output[1][1] == -0.1469999999999998 );

    INFO ("then test to see if the expected number of non-zeroes were produced");
    int nonzero_count {};
    for (auto& s : output[0]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    for (auto& s : output[1]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    for (auto& s : output[2]) {
        if (s != 0.0)
            ++nonzero_count;
    }
    REQUIRE( nonzero_count == 4 );
}


// NW: developed in response to issue here: https://github.com/Cycling74/min-lib/issues/22
TEST_CASE ("Clear message should reset values in memory to 0.0, but preserve size of memory allocation") {

    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 6 samples, vector-size of 4 samples");

    sample_vector zero(4, 0.0);
    sample_vector impulse(4, 0.0);
    impulse[0] = 1.0;

    delay my_delay { 6 };

    INFO ("We process impulse, then two empty vectors of audio...");
    sample_vector output[3];
    for (auto& s : impulse)
        output[0].push_back( my_delay(s) );
    for (auto& s : zero)
        output[1].push_back( my_delay(s) );
    for (auto& s : zero)
        output[2].push_back( my_delay(s) );

    REQUIRE( output[0][0] == 0.0 ); // delay 0 samples
    REQUIRE( output[0][1] == 0.0 ); // delay 1 sample
    REQUIRE( output[0][2] == 0.0 ); // delay 2 samples
    REQUIRE( output[0][3] == 0.0 ); // ...

    REQUIRE( output[1][0] == 0.0 );
    REQUIRE( output[1][1] == 0.0 );
    REQUIRE( output[1][2] == 1.0 ); // delay 6 samples
    REQUIRE( output[1][3] == 0.0 );

    REQUIRE( output[2][0] == 0.0 );
    REQUIRE( output[2][1] == 0.0 );
    REQUIRE( output[2][2] == 0.0 );
    REQUIRE( output[2][3] == 0.0 );

    INFO ("Next we process impulse, but clear before the empty vectors...");
    sample_vector output2[3];
    for (auto& s : impulse)
        output2[0].push_back( my_delay(s) );

    my_delay.clear();

    for (auto& s : zero)
        output2[1].push_back( my_delay(s) );
    for (auto& s : zero)
        output2[2].push_back( my_delay(s) );

    REQUIRE( output2[0][0] == 0.0 ); // delay 0 samples
    REQUIRE( output2[0][1] == 0.0 ); // delay 1 sample
    REQUIRE( output2[0][2] == 0.0 ); // delay 2 samples
    REQUIRE( output2[0][3] == 0.0 ); // ...

    REQUIRE( output2[1][0] == 0.0 );
    REQUIRE( output2[1][1] == 0.0 );
    REQUIRE( output2[1][2] == 0.0 ); // delayed impulse should be cleared
    REQUIRE( output2[1][3] == 0.0 );

    REQUIRE( output2[2][0] == 0.0 );
    REQUIRE( output2[2][1] == 0.0 );
    REQUIRE( output2[2][2] == 0.0 );
    REQUIRE( output2[2][3] == 0.0 );

}

TEST_CASE ("Using tail() and write() functions") {
    using namespace c74::min;
    using namespace c74::min::lib;
    INFO ("Using a delay of 4 samples to process first vector of 16 samples");

    sample_vector					input		{ 0,1,0,0,		0,0,2,0,	3,0,0,0,	0,0,0,0 };
    sample_vector					output;
    sample_vector					reference	{ 0,0,0,0,		0,1,0,0,	0,0,2,0,	3,0,0,0 };
    delay	my_delay(4);

    INFO ("We process 1 vector of audio...");
    for (auto& s : input) {
        my_delay.write(s);
        output.push_back( my_delay.tail() );
    }

    REQUIRE_VECTOR_APPROX( output , reference );

    INFO ("Using an offset of 2 brings delay to (4 - 2) samples when we process second vector of 16 samples");
    sample_vector					output2;
    sample_vector					reference2	{ 0,0,0,1,		0,0,0,0,	2,0,3,0,	0,0,0,0 };

    INFO ("We process second vector of audio...");
    for (auto& s : input) {
        my_delay.write(s);
        output2.push_back( my_delay.tail(2) );
    }

    REQUIRE_VECTOR_APPROX( output2 , reference2 );

    INFO ("Changing delay to 2.1 samples (requires interpolation) when we process third vector of 16 samples");
    sample_vector					output3;
    sample_vector					reference3	{ 0,0,0,0.9,	0.1,0,0,0,	1.8,0.2,2.7,0.3, 0,0,0,0	};

    my_delay.size(2.1);
    my_delay.change_interpolation(interpolator::type::linear);

    INFO ("We process third vector of audio...");
    for (auto& s : input) {
        my_delay.write(s);
        output3.push_back( my_delay.tail() );
    }

    REQUIRE_VECTOR_APPROX( output3 , reference3 );

}

