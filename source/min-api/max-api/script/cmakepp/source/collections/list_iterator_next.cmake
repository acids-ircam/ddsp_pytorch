## advances the iterator specified 
## and returns true if it is on a valid element (else false)
## sets the fields 
## ${it_ref}.index
## ${it_ref}.length
## ${it_ref}.list_ref
## ${it_ref}.value (only if a valid value exists)
function(list_iterator_next it_ref)
  list(GET ${it_ref} 0 list_ref)
  list(GET ${it_ref} 1 length)
  list(GET ${it_ref} 2 index)
  math(EXPR index "${index} + 1")    
  #print_vars(list_ref length index)
  set(${it_ref} ${list_ref} ${length} ${index} PARENT_SCOPE)
  set(${it_ref}.index ${index} PARENT_SCOPE)
  set(${it_ref}.length ${length} PARENT_SCOPE)
  set(${it_ref}.list_ref ${list_ref} PARENT_SCOPE)
  if(${index} LESS ${length})
    list(GET ${list_ref} ${index} value)
    set(${it_ref}.value "${value}" PARENT_SCOPE)
    return(true)
  else()
    set(${it_ref}.value PARENT_SCOPE)
    return(false)
  endif()
endfunction()
