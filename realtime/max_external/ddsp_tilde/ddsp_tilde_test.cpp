#include "c74_min_unittest.h"
#include "ddsp_tilde.cpp"

SCENARIO("object behave correctly") {
  ext_main(nullptr);
  GIVEN("an instance of ddsp~") {
    test_wrapper<ddsp_tilde> instance;
    ddsp_tilde &my_object = instance;

    WHEN("the default are used") {
      THEN("the output is halved") {
        auto result = my_object(1.0);
        REQUIRE(result == Approx(.5));
      }
    }
  }
}