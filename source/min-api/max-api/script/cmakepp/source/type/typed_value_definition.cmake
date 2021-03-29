##
## parses the specified typed value definition
##
function(typed_value_definition definition)
  regex_cmake()
  set(regex "^([<\\[])(${regex_cmake_flag})(\\{(.*)\\})?(=>(${regex_cmake_identifier}))?(:(.*))?(\\]|>)$")
  if("${definition}" MATCHES "${regex}")   
    set(name "${CMAKE_MATCH_2}")

    if(CMAKE_MATCH_6)
      set(variable_name "${CMAKE_MATCH_6}")    
    else()
      set(variable_name "${CMAKE_MATCH_2}")
    endif()
    set(type_def "${CMAKE_MATCH_8}")
    set(comment "${CMAKE_MATCH_4}")


    if("${CMAKE_MATCH_1}" STREQUAL "<")
      set(kind positional)
    else()
      set(kind nonpositional)
    endif()

    string(REGEX REPLACE "^\"(.*)\"$" "\\1" comment "${comment}")

    set(type)
    set(optional false)
    set(default_value)
    
    if(type_def)
      if("${type_def}" MATCHES "<(${regex_cmake_identifier})(.*)>(.*)")      
        set(type "${CMAKE_MATCH_1}")
        if("${CMAKE_MATCH_3}_" STREQUAL "?_")        
          set(optional true)      
        elseif("${CMAKE_MATCH_3}" MATCHES "^=(.*)")
          set(default_value "${CMAKE_MATCH_1}")
          string(REGEX REPLACE "^\"(.*)\"$" "\\1" default_value "${default_value}")          
        endif()      
      else() 
        message(FATAL_ERROR "invalid type definition for '${definition}': '${__letsv_type}' (needs to be inside angular brackets)")
      endif()
    endif()
    ## replace variable name
    string(REGEX REPLACE "--(.*)" "\\1" variable_name "${variable_name}")
    string(REPLACE "-" "_" variable_name "${variable_name}")

    map_capture_new(name variable_name type_def comment kind type optional default_value)
    return_ans()
  else()
    message(FATAL_ERROR "invalid definition: ${definition}")
  endif()
endfunction()