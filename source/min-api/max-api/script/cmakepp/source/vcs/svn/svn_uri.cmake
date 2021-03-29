## returns the svn_uri for the given ARGN
## if its empty emtpy is returned
## if it exists it is returned
## if it exists after qualification the qualifed path is returned
## else it is retunred
function(svn_uri)

  set(uri ${ARGN})
  if(NOT uri)
    return()
  endif()
  if(EXISTS "${uri}")
    return("${uri}")
  endif()
  path("${uri}")
  ans(uri_path)
  if(EXISTS "${uri_path}")
    return_ref(uri_path)
  endif()
  return_ref(uri)
endfunction()


