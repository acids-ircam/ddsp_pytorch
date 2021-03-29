

function(file_data_get dir id)
  set(nav ${ARGN})
  file_data_read("${dir}" "${id}")
  ans(res)
  if("${nav}_" STREQUAL "_" OR "${nav}_" STREQUAL "._")
    return_ref(res)
  endif()
  nav(data = "res.${nav}")
  return_ref(data)
endfunction()