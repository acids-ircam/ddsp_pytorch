
# extracts all flags specified and returns a map with the key being the flag name if it was found and the value being set to tru
# e.g. list_extract_flags([a,b,c,d] a c e) -> {a:true,c:true}, [b,d]
function(list_extract_flags __list_extract_flags_lst)
  list_find_flags("${__list_extract_flags_lst}" ${ARGN})
  ans(__list_extract_flags_flag_map)
  map_keys(${__list_extract_flags_flag_map})
  ans(__list_extract_flags_found_flags)
  list_remove("${__list_extract_flags_lst}" ${__list_extract_flags_found_flags})
 # list(REMOVE_ITEM "${__list_extract_flags_lst}" ${__list_extract_flags_found_flags})
  set("${__list_extract_flags_lst}" ${${__list_extract_flags_lst}} PARENT_SCOPE)
  return(${__list_extract_flags_flag_map})
endfunction()
