##
## extracts a single typed value defined by def
##
function(list_extract_defined_value lst def)
  map_tryget(${def} kind)
  ans(kind)
  map_tryget("${def}" type)
  ans(type)
  set(value)
  if("${kind}" STREQUAL "nonpositional")  
    map_tryget("${def}" name)
    ans(name)
    if(NOT type)
      list_extract_flag(${lst} ${name})
      ans(value)
    else()
      list_extract_labelled_value(${lst} ${name})
      ans(value)
    endif()    
  else()
    list_pop_front(${lst})
    ans(value)
  endif()

  encoded_list_decode("${value}")
  ans(value)

  if("${value}_" STREQUAL "_")
    map_tryget("${def}" default_value)
    ans(value)    
  endif()

  map_tryget("${def}" optional)
  ans(optional)

  if(NOT optional AND NOT "${value}_" STREQUAL "_" )
    if(type AND NOT "${type}" MATCHES "^(any)|(string)$" AND COMMAND "t_${type}")  
      eval("t_${type}(\"${value}\")")
      ans_extract(success)
      ans(value)

      if(NOT success)
        message(FATAL_ERROR "could not parse ${type} from '${value}'")
      endif()
    endif()
  else()
    ## optional
  endif()
  
  set(${__lst} ${${__lst}} PARENT_SCOPE)
  return_ref(value)
endfunction()