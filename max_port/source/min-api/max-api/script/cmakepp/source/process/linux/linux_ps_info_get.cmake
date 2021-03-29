
function(linux_ps_info_get pid)
  map_new()
  ans(map)
  linux_ps_info_capture("${pid}" "${map}" ${ARGN})
  return("${map}")

endfunction()
