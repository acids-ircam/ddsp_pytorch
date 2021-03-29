 function(semver_format version)
  semver_normalize("${version}")
  ans(version)

  #format("{version.major}.{version.minor}.{version.patch}")
  #ans(res)
  map_tryget(${version} major)
  ans(major)
  map_tryget(${version} minor)
  ans(minor)
  map_tryget(${version} patch)
  ans(patch)
  set(res "${major}.${minor}.${patch}")

  map_tryget("${version}" prerelease)
  ans(prerelease)
  if(NOT "${prerelease}_" STREQUAL "_")
    set(res "${res}-${prerelease}")
  endif()

  map_tryget("${version}" metadata)
  ans(metadata)
  if(NOT "${metadata}_" STREQUAL "_")
    set(res "${res}+${metadata}")
  endif()

  return_ref(res)

 endfunction()