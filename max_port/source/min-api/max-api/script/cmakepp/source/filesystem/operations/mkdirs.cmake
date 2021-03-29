# creates all specified dirs
function(mkdirs)
  set(res)
  foreach(path ${ARGN})
    mkdir("${path}")
    ans(p)
    list(APPEND res "${p}")    
  endforeach()
  return_ref(res)
endfunction()


