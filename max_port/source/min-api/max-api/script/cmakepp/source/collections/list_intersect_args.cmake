# returns only those flags which are contained in list and in the varargs
# ie list = [--a --b --c --d]
# list_intersect_args(list --c --d --e) ->  [--c --d]
function(list_intersect_args __list_intersect_args_lst)
  set(__list_intersect_args_flags ${ARGN})
  list_intersect(${__list_intersect_args_lst} __list_intersect_args_flags)
  return_ans()
endfunction()