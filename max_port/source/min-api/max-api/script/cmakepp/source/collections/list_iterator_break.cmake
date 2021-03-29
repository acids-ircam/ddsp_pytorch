## advances the iterator using list_iterator_next 
## and breaks the current loop when the iterator is done
macro(list_iterator_break it_ref)
  list_iterator_next(${it_ref})
  if(NOT __ans)
    break()
  endif()
endmacro()