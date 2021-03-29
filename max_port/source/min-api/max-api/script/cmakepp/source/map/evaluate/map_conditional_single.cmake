
  function(map_conditional_single parameters)
      set(value ${ARGN})

      
      is_map(${value})
      ans(is_map)

      if(NOT is_map)
      
        return_ref(value)
      endif()
     
      
      map_keys(${value})
      ans(keys)


      list_peek_front(keys)
      ans(firstKey)


      if(firstKey MATCHES "^\\$(.*)$")
        set(type "${CMAKE_MATCH_1}")
        if(COMMAND "map_conditional_${type}")
          #message("map_conditional_${type}(\"\${parameters}\" \"\${value}\")")
          eval("map_conditional_${type}(\"\${parameters}\" \"\${value}\")")
          ans(value)
        else()
          map_conditional_default("${parameters}" "${value}")
          ans(value)
        endif()        
      else()
          map_conditional_default("${parameters}" "${value}")
          ans(value)      
      endif()
      

      return_ref(value)
  endfunction()
