

function(map_navigate_set_if_missing navigation_expr)
  map_navigate(result ${navigation_expr})
  if(NOT result OR "${result}" STREQUAL "${navigation_expr}")
    map_navigate_set("${navigation_expr}" ${ARGN})
  endif() 
endfunction()