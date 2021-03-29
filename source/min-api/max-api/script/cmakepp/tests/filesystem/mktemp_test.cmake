function(test)
  mktemp()
  ans(p)

  assert(IS_DIRECTORY "${p}")

  mktemp()
  ans(p2)

  assert(IS_DIRECTORY "${p2}")

  assert(NOT "${p}" STREQUAL "${p2}")

  mktemp("${test_dir}")
  ans(p3)

  assert(IS_DIRECTORY "${p3}")
  assert("${p3}" MATCHES "^${test_dir}.*")

endfunction()