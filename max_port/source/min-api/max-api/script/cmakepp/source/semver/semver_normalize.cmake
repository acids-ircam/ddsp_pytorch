# returns a normalized version for a string or a object
# sets all missing version numbers to 0
# even an empty string is transformed to a version: it will be version 0.0.0 
function(semver_normalize version)
  semver("${version}")
  ans(version)

  if(NOT version)
    semver("0.0.0")
    ans(version)
  endif()

  nav(version.major)
  ans(current)
  if(NOT current)
    nav(version.major 0)
  endif() 


  nav(version.minor)
  ans(current)
  if(NOT current)
    nav(version.minor 0)
  endif() 


  nav(version.patch)
  ans(current)
  if(NOT current)
    nav(version.patch 0)
  endif() 

  return(${version})
endfunction()