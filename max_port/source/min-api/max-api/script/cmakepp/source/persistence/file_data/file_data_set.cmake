

function(file_data_set dir id nav)
  set(args ${ARGN})

  if("${nav}" STREQUAL "." OR "${nav}_" STREQUAL "_")
    file_data_write("${dir}" "${id}" ${ARGN})
    return_ans()
  endif()
  file_data_read("${dir}" "${id}")
  ans(res)
  map_navigate_set("res.${nav}" ${ARGN})
  file_data_write("${dir}" "${id}" ${res})
  return_ans()
endfunction()


   
  