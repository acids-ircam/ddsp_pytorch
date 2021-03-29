## `(<glob ignore expression...> [--relative] [--recurse]) -> <path...>`
##
## 
function(glob_ignore)
  set(args ${ARGN})
  list_extract_flag_name(args --relative)
  ans(relative)
  list_extract_flag_name(args --recurse)
  ans(recurse)


  glob_expression_parse(${args})
  ans(glob_expression)

  map_import_properties(${glob_expression} include exclude)

  glob(${relative} ${include} ${recurse})
  ans(included_paths)

  glob(${relative} ${exclude} ${recurse})
  ans(excluded_paths)
  if(excluded_paths)
    list(REMOVE_ITEM included_paths ${excluded_paths})
  endif()
  return_ref(included_paths)
endfunction()

