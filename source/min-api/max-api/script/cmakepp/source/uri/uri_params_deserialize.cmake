
  function(uri_params_deserialize query)
      
    string(REPLACE "&" "\;" query_assignments "${query}")
    set(query_assignments ${query_assignments})
    string(ASCII 21 c)
    map_new()
    ans(query_data)
    foreach(query_assignment ${query_assignments})
      string(REPLACE "=" "\;"  value "${query_assignment}")
      set(value ${value})
      list_pop_front(value)
      ans(key)
      set(path "${key}")      

      string(REPLACE "[]" "${c}" path "${path}")      
      string(REGEX REPLACE "\\[([^0-9]+)\\]" ".\\1" path "${path}")
      string(REPLACE "${c}" "[]" path "${path}")


      uri_decode("${path}")
      ans(path)
      uri_decode("${value}")
      ans(value)  


      ref_nav_set("${query_data}" "!${path}" "${value}")

    endforeach()
    return_ref(query_data)
  endfunction()