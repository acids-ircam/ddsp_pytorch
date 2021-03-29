## this function moves the map iterator to the next position
## and returns true if it was possible
## e.g.
## map_iterator_next(myiterator) 
## ans(ok) ## is true if iterator had a next element
## variables ${myiterator.key} and ${myiterator.value} are available
macro(map_iterator_next it_ref)
  list(LENGTH "${it_ref}" __map_iterator_next_length)
  if("${__map_iterator_next_length}" GREATER 1)
    list(REMOVE_AT "${it_ref}" 1)
    if(NOT "${__map_iterator_next_length}" EQUAL 2)
      list(GET "${it_ref}" 1 "${it_ref}.key")
      list(GET "${it_ref}" 0 "__map_iterator_map")
      get_property("${it_ref}.value" GLOBAL PROPERTY "${__map_iterator_map}.${${it_ref}.key}")
      set(__ans true)
    else()
      set(__ans false)
      set("${it_ref}.end" true)
    endif() 
  else()
    set("${it_ref}.end" true)
    set(__ans false)
  endif()
endmacro()
