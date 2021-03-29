

function(assertf)
  set(args ${ARGN})
  list_extract_flag(args DEREF)
  assert(${args} DEREF)
  return()
endfunction()