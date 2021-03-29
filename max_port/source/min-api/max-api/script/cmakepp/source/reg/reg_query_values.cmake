

  ## returns a map contains all the values in the specified registry key
  function(reg_query_values key)
    reg_query("${key}")
    ans(entries)
    map_new()
    ans(res)
    foreach(entry ${entries})
      scope_import_map(${entry})
      if(NOT "${value}_" STREQUAL "_")        
        map_set("${res}" "${value_name}" "${value}")
      endif()
    endforeach()
    return_ref(res)
  endfunction()
