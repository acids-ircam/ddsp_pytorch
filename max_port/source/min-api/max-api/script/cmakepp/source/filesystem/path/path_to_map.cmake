
# writes the path the map creating submaps for every directory

function(path_to_map map path)
  
  path_split("${path}")
  ans(path_parts)

  set(current ${map})
  while(true)
    list_pop_front(path_parts)
    ans(current_part)


    
    map_tryget(${current} "${current_part}")
    ans(current_map)

    if(NOT path_parts)
      if(NOT current_map)
      map_set(${current} "${current_part}" "${path}")
      endif()
      return()
    endif()

    is_map("${current_map}")
    ans(ismap)

    if(NOT ismap)
      map_new()
      ans(current_map)
    endif()

    map_set(${current} "${current_part}" ${current_map})
    set(current ${current_map})
  endwhile()
endfunction()