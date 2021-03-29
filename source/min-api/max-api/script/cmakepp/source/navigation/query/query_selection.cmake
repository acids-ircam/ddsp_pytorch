## `(<query: { <<selector:<navigation expression>> : <query literal>>...  } > <any>)-><any>`
##
## selects values depending on the specified query
## example 
## ```
## assign(input_data = "{
##   a: 'hello world',
##   b: 'goodbye world',
##   c: {
##    d: [1,2,3,4,5]
##   } 
## }")
## assign(result = query_selection("{a:{regex:{matchall:'[^ ]+'}}" ${input_data}))
## assertf("{result.a}" EQUALS  "hello" "world")
##
## ```
function(query_selection query)
  obj("${query}")


  map_keys("${query}")
  ans(selectors)



  set(result)


  ## loop through all selectors
  foreach(selector ${selectors})
    map_tryget(${query} "${selector}")
    ans(literal)

    ## check to see if selector ends with [...] 
    ## which indicates that action should be performed
    ## foreach item 
    ## 
    set(target_property)

    if("${selector}" MATCHES "(.+)=>(.+)")
      set(selector "${CMAKE_MATCH_1}")
      set(target_property "${CMAKE_MATCH_2}")
    endif()

    if("${selector}" STREQUAL "$")
      set(selector)
      set(foreach_item false)
    elseif("${selector}" MATCHES "(.*)\\[.*\\]$")
      if(NOT "${selector}" MATCHES "\\[-?([0]|[1-9][0-9]*)\\]$")
        if(NOT target_property)
          set(target_property "${CMAKE_MATCH_1}")
        endif()
        set(foreach_item true)
      endif()
    else()
      set(foreach_item false)
    endif()

    if("${target_property}_" STREQUAL "_")
      set(target_property "${selector}")
    endif()
    if("${target_property}" STREQUAL "$")
      set(target_property)
    endif()


    ref_nav_get("${ARGN}" ${selector})
    ans(value)

    query_literal("${literal}" __query_literal)
    ans(success)

    if(success)
      set(selection)
      if(foreach_item)
        foreach(item ${value})
          __query_literal(${item})
          if(NOT "${__ans}_" STREQUAL "_" )
            list(APPEND selection ${__ans})
          endif()
        endforeach()
      else()
        __query_literal(${value})
        if(NOT "${__ans}_" STREQUAL "_" )
          list(APPEND selection ${__ans})
        endif()
      endif()
      ref_nav_set("${result}" "!${target_property}" ${selection})
      ans(result)
    endif()

  endforeach()
  return_ref(result)

endfunction()