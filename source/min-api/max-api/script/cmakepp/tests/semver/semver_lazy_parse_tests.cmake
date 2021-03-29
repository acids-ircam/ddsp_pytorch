function(test)

  semver_parse_lazy("")
  ans(version)
  assert(NOT version) 

  semver_parse_lazy("asd")
  ans(version)
  assert(NOT version)



  semver_parse_lazy("1")
  ans(version)
  assert(version)
  assert(DEREF "{version.major}" STREQUAL "1")

  semver_parse_lazy("1.0.0" )
  ans(version)
  assert(version)
  assert(DEREF "{version.major}" STREQUAL "1")
  assert(DEREF "{version.minor}" STREQUAL "0")
  assert(DEREF "{version.patch}" STREQUAL "0")


  semver_parse_lazy("1.0.0-pre")
  ans(version)
  assert(DEREF "{version.major}" STREQUAL "1")
  assert(DEREF "{version.minor}" STREQUAL "0")
  assert(DEREF "{version.patch}" STREQUAL "0")
  assert(DEREF "{version.prerelease}" STREQUAL "pre")
  assert(DEREF "{version.tags}" STREQUAL "pre")

  semver_parse_lazy("1.0.0-pre+meta")
  ans(version)
  assert(DEREF "{version.major}" STREQUAL "1")
  assert(DEREF "{version.minor}" STREQUAL "0")
  assert(DEREF "{version.patch}" STREQUAL "0")
  assert(DEREF "{version.prerelease}" STREQUAL "pre")
  assert(DEREF "{version.tags}" STREQUAL "pre")
  assert(DEREF "{version.metadata}" STREQUAL "meta")

  semver_parse_lazy("1.0.0-pre-1.lolasd.va+meta-232-basd.dasd-23.bad3")
  ans(version)
  assert(DEREF "{version.major}" STREQUAL "1")
  assert(DEREF "{version.minor}" STREQUAL "0")
  assert(DEREF "{version.patch}" STREQUAL "0")
  assert(DEREF "{version.prerelease}" STREQUAL "pre-1.lolasd.va")
  assert(DEREF EQUALS "{version.tags}" "pre-1" "lolasd" "va")
  assert(DEREF "{version.metadata}" STREQUAL "meta-232-basd.dasd-23.bad3")
  assert(DEREF EQUALS "{version.metadatas}"  "meta-232-basd" "dasd-23" "bad3")


  semver_parse_lazy("0.0.0")
  ans(version)
  assert(DEREF "{version.major}" STREQUAL "0")
  assert(DEREF "{version.minor}" STREQUAL "0")
  assert(DEREF "{version.patch}" STREQUAL "0")
  endfunction()