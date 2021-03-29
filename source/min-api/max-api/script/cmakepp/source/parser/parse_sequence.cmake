
  function(parse_sequence rstring) 
    # create a copy from rstring 
    address_get(${rstring})
    ans(str)
    address_set_new("${str}")
    ans(str)

    # get sequence definitions
    map_get(${definition} sequence)
    ans(sequence)

    map_keys(${sequence})
    ans(sequence_keys)

    function(eval_sequence_expression rstring key res_map expression set_map)
      is_map("${expression}")
      ans(ismap)

      if(ismap)
        map_new()
        ans(definition)

        map_set(${definition} "parser" "sequence")
        map_set(${definition} "sequence" "${expression}")
        
#        json_print(${definition})
        parse_sequence("${rstring}")
        ans(res)

        if("${res}_" STREQUAL "_")
          return(false)
        endif()

        map_set(${result_map} "${key}" ${res})
        map_set(${set_map} "${key}" true)
        return(true)

      endif()      



      #message("Expr ${expression}")
      if("${expression}" STREQUAL "?")
        return(true)
      endif()
      # static value
      if("${expression}" MATCHES "^@")
        string(SUBSTRING "${expression}" 1 -1 expression)
        map_set("${res_map}" "${key}" "${expression}")
        return(true)
      endif()
      
      # null coalescing
      if("${expression}" MATCHES "[^@]*\\|")
        string_split_at_first(left right "${expression}" "|")
        eval_sequence_expression("${rstring}" "${key}" "${res_map}" "${left}" "${set_map}")
        ans(success)
        if(success)
          return(true)
        endif()
       # message("parsing right")
        eval_sequence_expression("${rstring}" "${key}" "${res_map}" "${right}" "${set_map}")
        return_ans()
      endif()

      # ternary operator ? :
      if("${expression}" MATCHES "[a-zA-Z0-9_-]+\\?.+")
        string_split_at_first(left right "${expression}" "?")
        set(else)
        if(NOT "${right}" MATCHES "^@")
          string_split_at_first(right else "${right}" ":")
        endif()
        map_tryget(${set_map} "${left}")
        ans(has_value)
        if(has_value)
          eval_sequence_expression("${rstring}" "${key}" "${res_map}" "${right}" "${set_map}")
          ans(success)
          if(success)
            return(true)
          endif()
          return(false)
        elseif(DEFINED else)
          eval_sequence_expression("${rstring}" "${key}" "${res_map}" "${else}" "${set_map}")
          ans(success)
          if(success)
            return(true)
          endif()

          return(false)
        else()
          return(true)
        endif()

      endif() 



      set(ignore false)
      set(optional false)
      set(default)


      if("${expression}" MATCHES "^\\?")
        string(SUBSTRING "${expression}" 1 -1 expression)
        set(optional true)
      endif()
      if("${expression}" MATCHES "^/")
        string(SUBSTRING "${expression}" 1 -1 expression)
        set(ignore true)
      endif()


      parse_string("${rstring}" "${expression}")
      ans(res)

      list(LENGTH res len)


      if(${len} EQUAL 0 AND NOT optional)
        return(false)
      endif()

      if(NOT "${ignore}" AND DEFINED res)
   #     message("setting at ${key}")
        map_set("${res_map}" "${key}" "${res}")
      endif()
      
      if(NOT ${len} EQUAL 0)
        map_set(${set_map} "${key}" "true")

      endif()
      return(true)
    endfunction()

    # match every element in sequence
    map_new()
    ans(result_map)

    map_new()
    ans(set_map)


    foreach(sequence_key ${sequence_keys})

      map_tryget("${sequence}" "${sequence_key}")
      ans(sequence_id)

      eval_sequence_expression("${str}" "${sequence_key}" "${result_map}" "${sequence_id}" "${set_map}")
      ans(success)
      if(NOT success)
        return()
      endif()
    endforeach()




    # if every element was  found set rstring to rest of string
    address_get(${str})
    ans(str)
    address_set(${rstring} "${str}")

    # return result
    return_ref(result_map)
  endfunction()



#    foreach(sequence_key ${sequence_keys})
#
#      map_tryget("${sequence}" "${sequence_key}")
#      ans(sequence_id)
#
#      if("${sequence_id}" MATCHES "^@")
#        string(SUBSTRING "${sequence_id}" 1 -1 sequence_id)
#        map_set("${result_map}" "${sequence_key}" "${sequence_id}")
#     
#      else()
#        set(ignore false)
#        set(optional false)
#        if("${sequence_id}" MATCHES "^\\?")
#          string(SUBSTRING "${sequence_id}" 1 -1 sequence_id)
#          set(optional true)
#        endif()
#        if("${sequence_id}" MATCHES "^/")
#          string(SUBSTRING "${sequence_id}" 1 -1 sequence_id)
#          set(ignore true)
#        endif()
#
#
#        parse_string("${str}" "${sequence_id}")
#        ans(res)
#
#        list(LENGTH res len)
#
#
#        if(${len} EQUAL 0 AND NOT optional)
#          return()
#        endif()
#
#        if(NOT "${ignore}")
#          map_set("${result_map}" "${sequence_key}" "${res}")
#        endif()
#      endif()
#    endforeach()