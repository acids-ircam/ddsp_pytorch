  function(parse_any rstring)
    # get defintiions for any
    map_get(${definition} any)
    ans(any)

    is_address("${any}")
    ans(isref)
    if(isref)
      address_get(${any})
      ans(any)
    endif()
    # loop through defintions and take the first one that works
    foreach(def_id ${any})
      parse_string("${rstring}" "${def_id}")
      ans(res)

      list(LENGTH res len)
      if("${len}" GREATER 0)
        return_ref(res)
      endif()

    endforeach()

    # return nothing if nothing matched
    return()
  endfunction()
