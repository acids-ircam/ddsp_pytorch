## `(<start:<token>> <end:<token>> [<limit:<uint>>])-><token>...` 
##
## returns the tokens which match type
## maximum tokens retunred is limit if specified
function(cmake_token_range_find_by_type start end type)
  set(limit ${ARGN})
  set(current ${start})
  set(count 0)
  set(result)
  while(current AND NOT "${current}" STREQUAL "${end}")
    if(limit AND NOT ${count} LESS "${limit}")
      return_ref(result)
    endif()
    map_tryget(${current} type)
    ans(current_type)

    if("${current_type}" MATCHES "${type}")
      list(APPEND result "${current}")
      math(EXPR count "${count} + 1")
    endif()
    map_tryget(${current} next)
    ans(current)
  endwhile()  
  return_ref(result)
endfunction()