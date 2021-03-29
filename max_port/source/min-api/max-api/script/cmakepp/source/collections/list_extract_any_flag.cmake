
# extracts all of the specified flags and returns true if any of them were found
function(list_extract_any_flag __list_extract_any_flag_lst)
  list_extract_flags("${__list_extract_any_flag_lst}" ${ARGN})
  set("${__list_extract_any_flag_lst}" ${${__list_extract_any_flag_lst}} PARENT_SCOPE)
  ans(flag_map)
  map_keys(${flag_map})
  ans(found_keys)
  list(LENGTH found_keys len)
  if(${len} GREATER 0)
    return(true)
  endif()
  return(false)
endfunction()


