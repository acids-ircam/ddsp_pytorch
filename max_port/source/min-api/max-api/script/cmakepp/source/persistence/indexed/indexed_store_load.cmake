
function(indexed_store_load )
  set(key ${ARGN})
  this_get(store_dir)
  set(path "${store_dir}/${key}-${key}-${key}")
  if(NOT EXISTS "${path}")
    return()
  endif()
  cmake_read("${path}")
  return_ans()

endfunction()