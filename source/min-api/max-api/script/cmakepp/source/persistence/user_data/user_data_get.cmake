
## returns data (read from storage) for the current user which is identified by <id>
## if no navigation arg is specified then the root data is returned
## else a navigation expression can be specified which returns a specific VALUE
## see nav function
function(user_data_get id)
  set(nav ${ARGN})
  user_data_read("${id}")
  ans(res)
  if("${nav}_" STREQUAL "_" OR "${nav}_" STREQUAL "._")
    return_ref(res)
  endif()
  nav(data = "res.${nav}")
  return_ref(data)
endfunction()