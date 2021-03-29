## `(<list ref> <key:<string>>)-><any ....>`
##
## returns the elements after the specified key
function(list_after __lst __key)
  list(LENGTH ${__lst} __len)
  if(NOT __len)
    return()
  endif()
  list(FIND ${__lst} "${__key}" __idx)
  if(__idx LESS 0)
    return()
  endif()
  math(EXPR __idx "${__idx} + 1")
  list_split(__ __rhs ${__lst} ${__idx})
  return_ref(__rhs)
endfunction()
