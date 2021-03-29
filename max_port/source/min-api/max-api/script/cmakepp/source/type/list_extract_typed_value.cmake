##  
##
## extracts a `<typed_value>` from the specified list
## ```
##   <typed_value>: ("<"|"[") ["--"] <parameter-name> [ "=>" <argument-name> ] [":" "<" <type-name> ">" ["?"] [ "=" <any> ] ] (">"|"]")
## ```
## the formal definition is relatively hard to understand but using examples it should be easy
## 
## ```
##    set(list1 a --test c b)
##    set(list2 a b c)
##    list_extract_typed_value(list1 [--test])  #=> "test;true"  rest=>a c b
##    list_extract_typed_value(list1 [--test])  #=> ""
##    list_extract_typed_value(list1 [--test:<any>=4])  #=> "test;4"
##    
## ```
function(list_extract_typed_value 
  __lst 
  __letsv_def

  )
  regex_cmake()
  set(__letsv_regex "^([<\\[])(${regex_cmake_flag})(\\{(.*)\\})?(=>(${regex_cmake_identifier}))?(:(.*))?(\\]|>)$")
  if("${__letsv_def}" MATCHES "${__letsv_regex}")
    if(CMAKE_MATCH_6)
      set(__letsv_name ${CMAKE_MATCH_6})
    else()
      set(__letsv_name ${CMAKE_MATCH_2})
    endif()
    set(__letsv_type ${CMAKE_MATCH_8})
    #set(__letsv_comment ${CMAKE_MATCH_4})
    ## set comment if outvar was detected
    if(ARGN)        
      string(REPLACE "\"" "" CMAKE_MATCH_4 "${CMAKE_MATCH_4}")
      set("${ARGN}" "${CMAKE_MATCH_4}" PARENT_SCOPE)        
    endif()
    if("${CMAKE_MATCH_1}" STREQUAL "<")
      set(__letsv_positional true)
    else()
      set(__letsv_positional false)
    endif()

    string(REGEX REPLACE "--(.*)" "\\1" __letsv_identifier "${__letsv_name}")
    string(REPLACE "-" "_" __letsv_identifier "${__letsv_identifier}")

    if(NOT __letsv_type)

    elseif("${__letsv_type}" MATCHES "<(${regex_cmake_identifier})(.*)>(.*)")
      #_message("${__letsv_type} : 0 ${CMAKE_MATCH_0} 1  ${CMAKE_MATCH_1} 2 ${CMAKE_MATCH_2} 3 ${CMAKE_MATCH_3}")
      set(__letsv_type "${CMAKE_MATCH_1}")
      set(__letsv_optional false)
      set(__letsv_default_value)
      if("${CMAKE_MATCH_3}_" STREQUAL "?_")
        set(__letsv_optional true)
      elseif("${CMAKE_MATCH_3}" MATCHES "^=(.*)")
        set(__letsv_default_value ${CMAKE_MATCH_1})
      endif()


      #print_vars(__letsv_identifier __letsv_optional)
    else() 
      message(FATAL_ERROR "invalid __letsv_type __letsv_def: '${__letsv_type}' (needs to be inside angular brackets)")
    endif()


    if(NOT __letsv_positional AND NOT __letsv_type)
      list_extract_flag(${__lst} ${__letsv_name})
      ans(__letsv_value)
    elseif(NOT __letsv_positional)
      list_extract_labelled_value(${__lst} ${__letsv_name})
      ans(__letsv_value)
    else()
      list_pop_front(${__lst})
      ans(__letsv_value)
    endif()

    encoded_list_decode("${__letsv_value}")
    ans(__letsv_value)

    if("${__letsv_value}_" STREQUAL "_")
      set(__letsv_value ${__letsv_default_value})
    endif()

    if(NOT __letsv_optional AND NOT "${__letsv_value}_" STREQUAL "_" )
      if(__letsv_type AND NOT "${__letsv_type}" MATCHES "^(any)|(string)$" AND COMMAND "t_${__letsv_type}")  
        eval("t_${__letsv_type}(\"${__letsv_value}\")")
        ans_extract(__letsv_success)
        ans(__letsv_value_parsed)

        if(NOT __letsv_success)
          message(FATAL_ERROR "could not parse ${__letsv_type} from '${__letsv_value}'")
        endif()
        set(__letsv_value ${__letsv_value_parsed})

      endif()
    else()
      ## optional
    endif()
      


    set(__ans ${__letsv_identifier} ${__letsv_value} PARENT_SCOPE)
    set(${__lst} ${${__lst}} PARENT_SCOPE)
  else()


    message(FATAL_ERROR "invalid definition: ${__letsv_def}")
  endif()
endfunction()