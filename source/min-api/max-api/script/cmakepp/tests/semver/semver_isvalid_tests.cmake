function(test)

  semver_isvalid("0.0.1")
  ans(res)
  assert(res)

  semver_isvalid("0.1")
  ans(res)
  assert(NOT res)


  semver_isvalid("0")
  ans(res)
  assert(NOT res)


  semver_isvalid("0.0.0-tag1-1.tag2-2.tag3.tag.23+meta1-sd.asd")
  ans(res)
  assert(res)


endfunction()