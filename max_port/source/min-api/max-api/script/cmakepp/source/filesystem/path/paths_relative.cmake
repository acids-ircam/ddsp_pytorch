

function(paths_relative path_base)
  set(res)
  foreach(path ${ARGN})
    path_relative("${path_base}" "${path}")
    ans(c)
    list(APPEND res "${c}")
  endforeach()
  return_ref(res)
endfunction()