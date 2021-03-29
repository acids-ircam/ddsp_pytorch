

  ## queryies the registry for the specified key
  ## returns a list of entries containing all direct child elements
  function(reg_query key)
    string(REPLACE / \\ key "${key}")
    reg(query "${key}" --process-handle)
    ans(res)

    map_tryget(${res} stdout)
    ans(output)


    map_tryget(${res} exit_code)
    ans(error)

    if(error)
      return()
    endif()
    
    string_encode_semicolon("${output}")
    ans(output)
    string(REPLACE "\n" ";" lines ${output})

    set(entries)
    foreach(line ${lines})
      reg_entry_parse("${key}" "${line}")
      ans(res)
      if(res)
        list(APPEND entries ${res})
      endif()
    endforeach()

    return_ref(entries)
  endfunction()