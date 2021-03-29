## `(<list&> <key:<string>>)-><any ....>`
##
## returns the elements before key
function(list_before __lst __key)
  list(LENGTH ${__lst} __len)
  if(NOT __len)
    return()
  endif()
  list(FIND ${__lst} "${__key}" __idx)
  if(__idx LESS 0)
    return()
  endif()
  math(EXPR __idx "${__idx} + 1")
  list_split(__lhs __ ${__lst} ${__idx})
  return_ref(__lhs)
endfunction()
