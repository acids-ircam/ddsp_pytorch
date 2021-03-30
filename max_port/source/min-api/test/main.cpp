#define CATCH_CONFIG_MAIN

#include "c74_min_catch.h"

using namespace c74::min;

class TestObject : public object<TestObject> {};

TEST_CASE("Attribute - ranges", "[attribute]") {
	TestObject my_object;
	attribute<number, threadsafe::no, limit::clamp> my_attr {&my_object, "My Attribute", 0.0, range {-10.0, 10.0} };
	
	SECTION("Cannot set attribute to a value outside its range") {
		const auto value = GENERATE(-100.0, 25.0, 11.5);
		my_attr = value;
		REQUIRE(static_cast<number>(my_attr) >= -10.0);
		REQUIRE(static_cast<number>(my_attr) <= 10.0);
	}
	
	SECTION("Setting an attribute's range changes its value") {
		const auto new_value = GENERATE(-5.0, 8.0);
		const auto new_minimum = GENERATE(1.0, 5.0);
		my_attr = new_value;
		my_attr.set_range({new_minimum, 10.0});
		REQUIRE(static_cast<number>(my_attr) == std::max(new_minimum, new_value));
	}
}

TEST_CASE("Attribute - repetitions", "[attribute]") {
	TestObject my_object;
	attribute<number, threadsafe::no, limit::clamp, allow_repetitions::no> my_attr {&my_object, "My Attribute", 0.0, range {-10.0, 10.0}};

	SECTION("Filtering out repetitions does not filter out a value when the range has changed") {
		my_attr = 5.0;
		my_attr.set_range({7.5, 10.0});

		REQUIRE(static_cast<number>(my_attr) == 7.5);
	}
}
