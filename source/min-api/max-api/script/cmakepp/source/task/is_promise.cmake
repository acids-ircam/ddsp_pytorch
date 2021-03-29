## `(<any>)-><bool>`
##
## returns true if the specified argument is a promise
function(is_promise)
  map_get_special("${ARGN}" $type)
  ans(type)
  if("${type}_" STREQUAL "promise_")
    set(__ans true PARENT_SCOPE)
  else()
    set(__ans false PARENT_SCOPE)
  endif()
endfunction()