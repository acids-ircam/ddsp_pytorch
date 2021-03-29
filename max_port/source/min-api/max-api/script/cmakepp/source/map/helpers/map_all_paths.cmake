


# returns all possible paths for the map
# (currently crashing on cycles cycles)
# todo: implement
function(map_all_paths)
  message(FATAL_ERROR "map_all_paths is not implemented yet")

  function(_map_all_paths event)
    if("${event}" STREQUAL "map_element_begin")
      address_get(${current_path})
      ans(current_path)
      set(cu)
    endif()
    if("${event}" STREQUAL "value")
      address_new(${})
    endif()
  endfunction()

  address_new()
  ans(current_path)
  address_new()
  ans(path_list)

  dfs_callback(_map_all_paths ${ARGN})
endfunction()