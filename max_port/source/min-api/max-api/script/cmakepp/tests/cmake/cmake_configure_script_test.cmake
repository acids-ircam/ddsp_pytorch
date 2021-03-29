function(test)


  cmake_configure_script("return(\${CMAKE_CURRENT_BINARY_DIR})")
  ans(res)

  assert(NOT EXISTS "${res}")


  cmake_configure_script("return(\${CMAKE_CURRENT_BINARY_DIR})" --target-dir "here")
  ans(res)

  assert("${res}" STREQUAL "${test_dir}/here/build")

endfunction()