function(test)


  semver_normalize("1")
  ans(res)
  semver_format(${res})
  ans(res)
  assert("${res}" STREQUAL "1.0.0")

  semver_normalize("")
  ans(res)
  semver_format(${res})
  ans(res)
  assert("${res}" STREQUAL "0.0.0")

  semver_normalize("1.0")
  ans(res)
  semver_format(${res})
  ans(res)
  assert("${res}" STREQUAL "1.0.0")

endfunction()