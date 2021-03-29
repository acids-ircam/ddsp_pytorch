## `(<start:<token>> <token type> [<value:<regex>>])-><token>`  
##
## returns the next token that has the specified token type
## or null
function(cmake_token_range_find_next_by_type range type)
  list_extract(range current end)
  set(regex ${ARGN})
  while(current AND NOT "${current}" STREQUAL "${end}")
    map_tryget(${current} type)
    ans(current_type)
    if("${current_type}" MATCHES "${type}")
      if(regex)
        map_tryget(${current} literal_value)
        ans(current_value)
        if("${current_value}" MATCHES "${regex}")
     #   print_vars(current_value regex match)
          return_ref(current)
        endif()
     #   print_vars(current_value regex nomatch)
      else()
        return_ref(current)
      endif()
    endif()
    map_tryget(${current} next)
    ans(current)
  endwhile()
endfunction()

