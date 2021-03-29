
function(parse_object rstring)
  
    # create a copy from rstring 
    address_get(${rstring})
    ans(str)
    address_set_new("${str}")
    ans(str)

    # get definitions
    map_tryget(${definition} begin)
    ans(begin_id)

    map_tryget(${definition} end)
    ans(end_id)
    
    map_tryget(${definition} keyvalue)
    ans(keyvalue_id)

    map_tryget(${definition} separator)
    ans(separator_id)         

    if(begin_id)
      parse_string(${str} ${begin_id})
      ans(res)
      list(LENGTH res len)
      if(${len} EQUAL 0)
        return()
      endif()
    endif()

    map_new()
    ans(result_object)

    set(has_result)

    while(true)
      # try to parse end of list if it was parsed stop iterating
      if(end_id)
        parse_string(${str} "${end_id}")
        ans(res)

        list(LENGTH res len)
        if(${len} GREATER 0)
          break()
        endif()
      endif()

      if(separator_id)
        if(has_result)
          parse_string(${str} "${separator_id}")
          ans(res)
          list(LENGTH res len)
          if(${len} EQUAL 0)
            if(NOT end)
              break()
            endif()
            return()
          endif()
        endif()
      endif()

      parse_string(${str} "${keyvalue_id}")
      ans(keyvalue)

      if(NOT keyvalue)
        if(NOT end)
          break()
        endif()
        return()
      endif()

      map_get(${keyvalue} key)
      ans(object_key)

      map_get(${keyvalue} value)
      ans(object_value)

      if(NOT has_result)
        set(has_result true)
      endif()

      if("${object_value}_" STREQUAL "_")
        
        set(object_value "")
      endif()
      
      map_set("${result_object}" "${object_key}" "${object_value}")

    endwhile()    


    # if every element was  found set rstring to rest of string
    address_get(${str})
    ans(str)
    address_set(${rstring} "${str}")

    # return result
    return_ref(result_object)
endfunction()