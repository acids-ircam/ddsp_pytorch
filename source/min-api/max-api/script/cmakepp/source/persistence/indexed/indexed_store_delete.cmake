
function(indexed_store_delete)
  set(key ${ARGN})
  this_get(store_dir)
  file(GLOB files "${store_dir}/*${key}*")
  if(NOT files)
    return(false)
  endif()
  file(REMOVE ${files})
  return(true)
endfunction()


