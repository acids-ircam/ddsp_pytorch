
  ## glob_paths(<unqualified glob path...>) -> <qualified glob path...>
  ##
  ## 
 function(glob_paths)
  set(result)
  foreach(path ${ARGN})
    glob_path(${path})
    ans(res)
    list(APPEND result ${res})
  endforeach()
  return_ref(result)
 endfunction()
