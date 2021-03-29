
## map_remove_item
##
## removes the specified items from <map>.<prop>
## returns the number of items removed
function(map_remove_item map prop)
  map_tryget("${map}" "${prop}")
  ans(vals)
  list_remove(vals ${ARGN})
  ans(res)
  if(res)
    map_set_hidden("${map}" "${prop}" "${vals}")
  endif()
  return_ref(res)
endfunction()