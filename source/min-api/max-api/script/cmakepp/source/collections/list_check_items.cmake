## `(<list&> <query...>)-><bool>`
##  
## `<query> := <value>|'!'<value>|<value>'?'`
## 
## * checks to see that every value specified is contained in the list 
## * if the value is preceded by a `!` checks that the value is not in the list
## * if the value is succeeded by a `?` the value may or may not be contained
##
## returns true if all queries match
## 
function(list_check_items __lst)
  set(lst ${${__lst}})
  set(result 0)
  list(LENGTH ARGN len)

  foreach(item ${ARGN})
    set(negate false)
    set(optional false)
    if("${item}" MATCHES "^!(.+)$")
      set(item "${CMAKE_MATCH_1}")
      set(negate true)
    endif()
    if("${item}" MATCHES "^(.+)\\?$")
      set(item "${CMAKE_MATCH_1}")
      set(optional true)
    endif()

    list_contains(lst "${item}")
    ans(is_contained)

    if(false)
    elseif(    is_contained AND     negate AND     optional)
      list_remove(lst "${item}")
    elseif(    is_contained AND     negate AND NOT optional)
      return(false)
    elseif(    is_contained AND NOT negate AND     optional)
      list_remove(lst "${item}")
    elseif(    is_contained AND NOT negate AND NOT optional)
      list_remove(lst "${item}")
    elseif(NOT is_contained AND     negate AND     optional)
      list_remove(lst "${item}")
    elseif(NOT is_contained AND     negate AND NOT optional)
      list_remove(lst "${item}")
    elseif(NOT is_contained AND NOT negate AND     optional)
      list_remove(lst "${item}")
    elseif(NOT is_contained AND NOT negate AND NOT optional)
      return()
    endif()

   # print_vars(lst item is_contained negate optional)
  endforeach()

  list(LENGTH lst len)
  if(len)
    return(false)
  endif()
  return(true)
endfunction()