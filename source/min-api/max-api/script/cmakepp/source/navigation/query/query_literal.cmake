

function(query_literal)


  query_literal_definition_add(bool query_literal_bool "^((true)|(false))$")
  query_literal_definition_add(regex query_literal_regex "^/(.+)/$")
  query_literal_definition_add(gt query_literal_gt "^>([^=].*)")
  query_literal_definition_add(lt query_literal_lt "^<([^=].*)")
  query_literal_definition_add(eq query_literal_eq "^=([^=].*)")
  query_literal_definition_add(match query_literal_match "^\\?/(.+)/$" )
  query_literal_definition_add(strequal  query_literal_strequal "(.+)")  
  query_literal_definition_add(where query_literal_where "" )

    
  function(query_literal query_literal_instance )
    if("${query_literal_instance}_" STREQUAL "_")
      return()
    endif()

    is_address("${query_literal_instance}")
    ans(is_ref)

    if(is_ref)
      map_keys(${query_literal_instance})
      ans(type)
      query_literal_definition("${type}")
      ans(query_literal_definition)
      map_tryget(${query_literal_instance} "${type}")
      ans(query_literal_input)
    else()
      # is predicate?
      if(false)
        
      else()
        query_literal_definitions_with_regex()
        ans(definitions)
        foreach(def ${definitions})
          map_tryget(${def} regex)
          ans(regex)
          set(query_literal_input)
          if("${query_literal_instance}" MATCHES "${regex}")
            set(query_literal_input ${CMAKE_MATCH_1})
          endif()
        #   print_vars(query_literal_input query_literal_instance regex replace)
          if(NOT "${query_literal_input}_" STREQUAL "_")
            set(query_literal_definition ${def})
            break()
          endif()
        endforeach()

        # if("${query_literal_instance}" MATCHES "^(true)|(false)$")
        #   ## boolish
        #   map_new()
        #   ans(query_literal_definition)
        #   map_set(${query_literal_definition} bool ${query_literal_instance})
        # else()
        #   ## just a value -> strequal
        #   map_new()
        #   ans(query_literal_definition)
        #   map_set(${query_literal_definition} strequal ${query_literal_instance})
        # endif()
      endif()
    endif()
    if(NOT query_literal_definition)
      message(FATAL_ERROR "invalid query literal")
    endif()

    map_tryget(${query_literal_definition} function)
    ans(query_literal_function)

    if("${ARGN}_" STREQUAL "_")
      function_new()
      ans(alias)
    else()
      set(alias ${ARGN})
    endif()

    ## create a curried function
    eval( "
    function(${alias})
      ${query_literal_function}(\"${query_literal_input}\" \${ARGN})
      set(__ans \${__ans} PARENT_SCOPE)
    endfunction()
    ")
    return_ref(alias)
  endfunction()

  query_literal(${ARGN})
  return_ans()
endfunction()