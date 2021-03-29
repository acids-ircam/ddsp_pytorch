function(test)

  find_package(Subversion)
  if(NOT SUBVERSION_FOUND)
    message("test inconclusive - subversion not isntalled")
    return()
  endif()

  
  svn(--version --quiet --process-handle)  
  ans(res)

  json_print(${res})


  map_tryget(${res} stdout)
  ans(res)
  string(STRIP "${res}" res)

  assert("${res}" MATCHES "${Subversion_VERSION_SVN}")


endfunction()



