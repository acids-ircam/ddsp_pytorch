## 
##
## returns all elements whose index are specfied
## 
function(list_at __list_at_lst)
  set(__list_at_result)
  foreach(__list_at_idx ${ARGN})
    list_get(${__list_at_lst} ${__list_at_idx})
    list(APPEND __list_at_result ${__ans})
  endforeach()
  return_ref(__list_at_result)
endfunction()