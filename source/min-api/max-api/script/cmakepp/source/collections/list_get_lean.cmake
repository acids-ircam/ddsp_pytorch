
## quickly gets the items from the specified list
macro(list_get_lean __lst_ref)
  list(LENGTH ARGN __len)
  if(__len)
    list(GET "${__lst_ref}" ${ARGN})
  else()
    set(__ans)
  endif() 
endmacro()

