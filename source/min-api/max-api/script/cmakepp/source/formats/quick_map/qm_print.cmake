function(qm_print)
  qm_serialize(${ARGN})
  ans(res)

  message("${res}")
  return()
endfunction()