/// @file
///	@brief 		Unit test for the circular storage container class
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#define CATCH_CONFIG_MAIN
#include "c74_min_catch.h"


TEST_CASE ("Basic usage of the Circular Storage class") {
    INFO ("Using an 8-sample circular buffer")
    c74::min::lib::circular_storage<c74::min::sample>	circ(8);	// 8 samples
    c74::min::sample_vector								samples = {1,2,3,4,5};
    c74::min::sample_vector								head_samples(3);
    c74::min::sample_vector								tail_samples(3);

    INFO("When writing 5 samples into the 8 sample buffer the first time")
    circ.write(samples);
    INFO("Those 5 samples occupy the first 5 slots and the other slots are still initialized to 0.");
    REQUIRE( circ.item(0) == 1 );
    REQUIRE( circ.item(1) == 2 );
    REQUIRE( circ.item(2) == 3 );
    REQUIRE( circ.item(3) == 4 );
    REQUIRE( circ.item(4) == 5 );
    REQUIRE( circ.item(5) == 0 );
    REQUIRE( circ.item(6) == 0 );
    REQUIRE( circ.item(7) == 0 );

    INFO("When writing 5 more samples into the 8 sample buffer")
    samples = {6,7,8,9,10};
    circ.write(samples);
    INFO("Those samples begin writing where the previous write left off, and they wrap around and overwrite the beginning (oldest) samples.");
    REQUIRE(  circ.item(0) == 9 );
    REQUIRE(  circ.item(1) == 10 );
    REQUIRE(  circ.item(2) == 3 );
    REQUIRE(  circ.item(3) == 4 );
    REQUIRE(  circ.item(4) == 5 );
    REQUIRE(  circ.item(5) == 6 );
    REQUIRE(  circ.item(6) == 7 );
    REQUIRE(  circ.item(7) == 8 );

    INFO("When reading 3 samples from the head with no previous reads")
    circ.head(head_samples);
    INFO("We get the most recently written 3 sample values");
    REQUIRE( head_samples[0] == 8 );
    REQUIRE( head_samples[1] == 9 );
    REQUIRE( head_samples[2] == 10 );

    INFO("When we then read 3 samples from the tail")
    circ.tail(tail_samples);
    INFO("We get the oldest 3 sample values that were written");
    REQUIRE( tail_samples[0] == 3 );
    REQUIRE( tail_samples[1] == 4 );
    REQUIRE( tail_samples[2] == 5 );

    INFO("When we read 3 more samples from the head")
    circ.head(head_samples);
    INFO("We get the most recently written 3 sample values, just as before -- nothing has changed");
    REQUIRE( head_samples[0] == 8 );
    REQUIRE( head_samples[1] == 9 );
    REQUIRE( head_samples[2] == 10 );

    INFO("And when we then read 3 more samples from the tail")
    circ.tail(tail_samples);
    INFO("We still get the oldest 3 sample values that were written -- nothing has changed");
    REQUIRE( tail_samples[0] == 3 );
    REQUIRE( tail_samples[1] == 4 );
    REQUIRE( tail_samples[2] == 5 );

    INFO("We then WRITE 3 more samples")
    samples = {20, 21, 22};
    circ.write(samples);

    INFO("And the internal storage is updated with the new items in the correct locations, overwriting the 3 oldest values");
    REQUIRE( circ.item(0) == 9 );
    REQUIRE( circ.item(1) == 10 );
    REQUIRE( circ.item(2) == 20 );
    REQUIRE( circ.item(3) == 21 );
    REQUIRE( circ.item(4) == 22 );
    REQUIRE( circ.item(5) == 6 );
    REQUIRE( circ.item(6) == 7 );
    REQUIRE( circ.item(7) == 8 );

    INFO("If we read 3 samples from the head now")
    circ.head(head_samples);
    INFO("We get the 3 sample values we just wrote");
    REQUIRE( head_samples[0] == 20 );
    REQUIRE( head_samples[1] == 21 );
    REQUIRE( head_samples[2] == 22 );

    INFO("And when we read 3 more samples from the tail")
    circ.tail(tail_samples);
    INFO("We now have advanced 3 samples in the buffer since the last time we read the tail, because 3 items were written since then.");
    REQUIRE( tail_samples[0] == 6 );
    REQUIRE( tail_samples[1] == 7 );
    REQUIRE( tail_samples[2] == 8 );

    INFO("If we then WRITE another 3 samples")
    samples = {100, 101, 102};
    circ.write(samples);
    INFO("The internal storage is updated with the new items in the correct locations, overwriting the 3 oldest values");
    REQUIRE( circ.item(0) == 9 );
    REQUIRE( circ.item(1) == 10 );
    REQUIRE( circ.item(2) == 20 );
    REQUIRE( circ.item(3) == 21 );
    REQUIRE( circ.item(4) == 22 );
    REQUIRE( circ.item(5) == 100 );
    REQUIRE( circ.item(6) == 101 );
    REQUIRE( circ.item(7) == 102 );

    INFO("When we read 3 samples from the head now")
    circ.head(head_samples);
    INFO("We get the 3 sample values we just wrote");
    REQUIRE( head_samples[0] == 100 );
    REQUIRE( head_samples[1] == 101 );
    REQUIRE( head_samples[2] == 102 );

    INFO("And when read 3 more samples from the tail")
    circ.tail(tail_samples);
    INFO("We now have advanced 3 samples in the buffer since the last time we read the tail, because 3 items were written since then.");
    REQUIRE( tail_samples[0] == 9 );
    REQUIRE( tail_samples[1] == 10 );
    REQUIRE( tail_samples[2] == 20 );

    INFO("If we read 5 samples from the head now, having not written since the last time")
    head_samples.resize(5);
    tail_samples.resize(5);
    circ.head(head_samples);
    INFO("We get the 5 newest sample values, which includes the same 3 we got the last time we read.");
    REQUIRE( head_samples[0] == 21 );
    REQUIRE( head_samples[1] == 22 );
    REQUIRE( head_samples[2] == 100 );
    REQUIRE( head_samples[3] == 101 );
    REQUIRE( head_samples[4] == 102 );


    INFO("And when we read 5 samples from the tail")
    circ.tail(tail_samples);
    INFO("We read the fove oldest sample values, which includes the same 3 oldest values we got the last time we read the tail.");
    REQUIRE( tail_samples[0] == 9 );
    REQUIRE( tail_samples[1] == 10 );
    REQUIRE( tail_samples[2] == 20 );
    REQUIRE( tail_samples[3] == 21 );
    REQUIRE( tail_samples[4] == 22 );
}


TEST_CASE ("Using Circular Storage as the basis for an Audio Delay") {
    INFO("Using a 16-sample circular buffer and a processing vector-size of 4")

    c74::min::lib::circular_storage<c74::min::sample>	circ(16);
    c74::min::sample_vector								samples(4);
    c74::min::sample_vector								output(4);

    INFO("The default is that delay will be the size of the capacity of the buffer (16 samples)");
    REQUIRE(circ.size() == 16);

    INFO("When Reading from the delay line the first time (first 4 vectors)")
    INFO("The first 16 samples (4 vectors) will be zero because we read from it before we write into it");

    circ.tail(output);
    REQUIRE( output[0] == 0 );
    REQUIRE( output[1] == 0 );
    REQUIRE( output[2] == 0 );
    REQUIRE( output[3] == 0 );
    samples = {1,2,3,4};
    circ.write(samples);

    circ.tail(output);
    REQUIRE( output[0] == 0 );
    REQUIRE( output[1] == 0 );
    REQUIRE( output[2] == 0 );
    REQUIRE( output[3] == 0 );
    samples = {5,6,7,8};
    circ.write(samples);

    circ.tail(output);
    REQUIRE( output[0] == 0 );
    REQUIRE( output[1] == 0 );
    REQUIRE( output[2] == 0 );
    REQUIRE( output[3] == 0 );
    samples = {9,10,11,12};
    circ.write(samples);

    circ.tail(output);
    REQUIRE( output[0] == 0 );
    REQUIRE( output[1] == 0 );
    REQUIRE( output[2] == 0 );
    REQUIRE( output[3] == 0 );
    samples = {13,14,15,16};
    circ.write(samples);

    INFO("When Reading from the delay line for the 17th sample")
    INFO("The tail should produce what happened 16 samples ago (our delay time is 16 samples): 1,2,3,4");

    circ.tail(output);
    REQUIRE( output[0] == 1 );
    REQUIRE( output[1] == 2 );
    REQUIRE( output[2] == 3 );
    REQUIRE( output[3] == 4 );
    samples = {17,18,19,20};
    circ.write(samples);

    circ.tail(output);
    REQUIRE( output[0] == 5 );
    REQUIRE( output[1] == 6 );
    REQUIRE( output[2] == 7 );
    REQUIRE( output[3] == 8 );
    samples = {21,22,23,24};
    circ.write(samples);

    INFO("We change the delay time from 16-samples to 10-samples")
    circ.resize(10);
    circ.tail(output);
    REQUIRE( circ.size() == 10 );
    REQUIRE( circ.capacity() == 16 );

    INFO("if tail produced what happened 10 samples ago it would be: 15,16,17,18 -- but we just shortened the delay, which means some samples are going to get dropped");
    REQUIRE( output[0] == 9 );
    REQUIRE( output[1] == 10 );
    REQUIRE( output[2] == 17 );
    REQUIRE( output[3] == 18 );
    samples = {25,26,27,28};
    circ.write(samples);

    INFO("When we process the next vector after the delay time change")
    INFO("tail produces what happened 10 samples ago: 19,20,21,22");
    circ.tail(output);
    REQUIRE( output[0] == 19 );
    REQUIRE( output[1] == 20 );
    REQUIRE( output[2] == 21 );
    REQUIRE( output[3] == 22 );
    samples = {29,30,31,32};
    circ.write(samples);
}

TEST_CASE ("Test zero() and clear() functions of Circular Storage") {
    INFO ("Using an 8-sample circular buffer")
    c74::min::lib::circular_storage<c74::min::sample>	circ(8);	// 8 samples
    c74::min::sample_vector								samples = {1,2,3,4,5,6,7,8};

    INFO("The default size will be the capacity of the circular buffer (8 samples)");
    REQUIRE( circ.size() == 8 );

    INFO("After we write in 8 values, the internal storage is updated with the new items in the correct locations");
    circ.write(samples);
    REQUIRE( circ.item(0) == 1 );
    REQUIRE( circ.item(1) == 2 );
    REQUIRE( circ.item(2) == 3 );
    REQUIRE( circ.item(3) == 4 );
    REQUIRE( circ.item(4) == 5 );
    REQUIRE( circ.item(5) == 6 );
    REQUIRE( circ.item(6) == 7 );
    REQUIRE( circ.item(7) == 8 );

    INFO("After we call zero(), the size of the internal storage in unchanged...");
    circ.zero();
    REQUIRE( circ.size() == 8 );

    INFO("...but all values have been set to zero");
    REQUIRE( circ.item(0) == 0 );
    REQUIRE( circ.item(1) == 0 );
    REQUIRE( circ.item(2) == 0 );
    REQUIRE( circ.item(3) == 0 );
    REQUIRE( circ.item(4) == 0 );
    REQUIRE( circ.item(5) == 0 );
    REQUIRE( circ.item(6) == 0 );
    REQUIRE( circ.item(7) == 0 );

    INFO("After we call clear(), the internal storage has a size of 0");
    circ.clear();
    REQUIRE( circ.size() == 0 );

}

// NW: Used the Audio Delay test above as the template for this new test
TEST_CASE ("Comparing outputs from two versions of tail() function to ensure they match") {
    INFO("Using a 16-sample circular buffer and a processing vector-size of 4")

    c74::min::lib::circular_storage<c74::min::sample>	circ(16);
    c74::min::sample_vector								samples(4);
    c74::min::sample_vector								output(4);

    INFO("The default is that delay will be the size of the capacity of the buffer (16 samples)");
    REQUIRE(circ.size() == 16);

    INFO("When Reading from the delay line the first time (first 4 vectors)")
    INFO("The first 16 samples (4 vectors) will be zero because we read from it before we write into it");

    circ.tail(output);
    REQUIRE( output[0] == 0 );
    REQUIRE( output[1] == 0 );
    REQUIRE( output[2] == 0 );
    REQUIRE( output[3] == 0 );
    samples = {1,2,3,4};
    circ.write(samples);

    circ.tail(output);
    REQUIRE( output[0] == circ.tail(0) );
    REQUIRE( output[1] == circ.tail(1) );
    REQUIRE( output[2] == circ.tail(2) );
    REQUIRE( output[3] == circ.tail(3) );
    samples = {5,6,7,8};
    circ.write(samples);

    circ.tail(output);
    REQUIRE( output[0] == circ.tail(0) );
    REQUIRE( output[1] == circ.tail(1) );
    REQUIRE( output[2] == circ.tail(2) );
    REQUIRE( output[3] == circ.tail(3) );
    samples = {9,10,11,12};
    circ.write(samples);

    circ.tail(output);
    REQUIRE( output[0] == circ.tail(0) );
    REQUIRE( output[1] == circ.tail(1) );
    REQUIRE( output[2] == circ.tail(2) );
    REQUIRE( output[3] == circ.tail(3) );
    samples = {13,14,15,16};
    circ.write(samples);

    INFO("When Reading from the delay line for the 17th sample")
    INFO("The tail should produce what happened 16 samples ago (our delay time is 16 samples): 1,2,3,4");

    circ.tail(output);
    REQUIRE( output[0] == circ.tail(0) );
    REQUIRE( output[1] == circ.tail(1) );
    REQUIRE( output[2] == circ.tail(2) );
    REQUIRE( output[3] == circ.tail(3) );
    samples = {17,18,19,20};
    circ.write(samples);

    circ.tail(output);
    REQUIRE( output[0] == circ.tail(0) );
    REQUIRE( output[1] == circ.tail(1) );
    REQUIRE( output[2] == circ.tail(2) );
    REQUIRE( output[3] == circ.tail(3) );
    samples = {21,22,23,24};
    circ.write(samples);

    INFO("We change the delay time from 16-samples to 10-samples")
    circ.resize(10);
    circ.tail(output);
    REQUIRE( circ.size() == 10 );
    REQUIRE( circ.capacity() == 16 );

    INFO("if tail produced what happened 10 samples ago it would be: 15,16,17,18 -- but we just shortened the delay, which means some samples are going to get dropped");
    REQUIRE( output[0] == circ.tail(0) );
    REQUIRE( output[1] == circ.tail(1) );
    REQUIRE( output[2] == circ.tail(2) );
    REQUIRE( output[3] == circ.tail(3) );
    samples = {25,26,27,28};
    circ.write(samples);

    INFO("When we process the next vector after the delay time change")
    INFO("tail produces what happened 10 samples ago: 19,20,21,22");
    circ.tail(output);
    REQUIRE( output[0] == circ.tail(0) );
    REQUIRE( output[1] == circ.tail(1) );
    REQUIRE( output[2] == circ.tail(2) );
    REQUIRE( output[3] == circ.tail(3) );
    samples = {29,30,31,32};
    circ.write(samples);
}
