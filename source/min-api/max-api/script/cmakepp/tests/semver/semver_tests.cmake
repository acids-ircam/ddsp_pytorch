function(semver_tests)

  semver("0.1")
  ans(res)
  assert(res)
  assert(DEREF "{res.major}" STREQUAL "0")
  assert(DEREF "{res.minor}" STREQUAL "1")
  

endfunction()