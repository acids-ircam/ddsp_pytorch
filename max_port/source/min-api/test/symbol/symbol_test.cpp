/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#define CATCH_CONFIG_MAIN
#include "c74_min_catch.h"


TEST_CASE( "Symbol Class", "[symbols]" ) {

    SECTION("symbol assignments") {
        c74::min::symbol s1 = "foo";
        const char* c1 = s1;

        REQUIRE( s1 == "foo" );
        REQUIRE( !strcmp(c1, "foo") );
    }

}
