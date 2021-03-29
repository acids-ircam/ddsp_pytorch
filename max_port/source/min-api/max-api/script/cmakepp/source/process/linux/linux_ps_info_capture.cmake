

function(linux_ps_info_capture pid map)

  foreach(key ${ARGN})
    linux_ps_info("${pid}" "${key}")
    ans(val)
    map_set(${map} "${key}" "${val}")

  endforeach()
  return()
endfunction()
