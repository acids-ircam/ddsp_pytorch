message(STATUS "running test for cmake functions")

# include cmakepp
include("${CMAKE_CURRENT_LIST_DIR}/../cmakepp.cmake")

## execute all tests in test directory
if("$ENV{CMAKEPP_TEST_EXECUTE_PARALLEL}_" STREQUAL "true_" )
  test_execute_glob_parallel(
    "${CMAKE_CURRENT_LIST_DIR}/../tests/**.cmake"
    --recurse --no-status)
else()
  test_execute_glob(
    "${CMAKE_CURRENT_LIST_DIR}/../tests/**.cmake"
    --recurse)
endif()

## beep three times to indicate end of testrun...
#beep()
#beep()
#beep()
