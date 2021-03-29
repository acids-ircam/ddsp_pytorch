
## list_split_at()
##
##
function(list_split_at lhs rhs __lst key)
  list(LENGTH ${__lst} len)
  if(NOT len)
    set(${lhs} PARENT_SCOPE)
    set(${rhs} PARENT_SCOPE)
    return()
  endif()

  list(FIND ${__lst} ${key} idx)

  list_split(${lhs} ${rhs} ${__lst} ${idx})

  set(${lhs} ${${lhs}} PARENT_SCOPE)
  set(${rhs} ${${rhs}} PARENT_SCOPE)

  return()
endfunction()