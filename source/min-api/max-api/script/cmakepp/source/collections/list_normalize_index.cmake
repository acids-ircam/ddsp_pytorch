# returns the normalized index.  negative indices are transformed to i => length - i
# if the index is out of range after transformation -1 is returned and a warnign is issued
# note: index evaluating to length are valid (one behind last)
function(list_normalize_index __lst index )
  set(idx ${index})
  list(LENGTH ${__lst} length)

  if("${idx}" STREQUAL "*")
    set(idx ${length})
  endif()
  
  if(${idx} LESS 0)
    math(EXPR idx "${length} ${idx} + 1")
  endif()
  if(${idx} LESS 0)
    message(WARNING "index out of range: ${index} (${idx}) length of list '${lst}': ${length}")
    return(-1)
  endif()

  if(${idx} GREATER ${length})
    message(WARNING "index out of range: ${index} (${idx}) length of list '${lst}': ${length}")
    return(-1)
  endif()
  return(${idx})
endfunction()