# uses the selector on each element of the list
function(list_select __list_select_lst selector)
  list(LENGTH ${__list_select_lst} l)
  message(list_select ${l})
  set(__list_select_result_list)

  foreach(item ${${__list_select_lst}})
		rcall(res = "${selector}"("${item}"))
		list(APPEND __list_select_result_list ${res})

	endforeach()
  message("list_select end")
	return_ref(__list_select_result_list)
endfunction()



## fast implementation of list_select
function(list_select __list_select_lst __list_select_selector)
  function_import("${__list_select_selector}" as __list_select_selector REDEFINE)

  set(__res)
  set(__ans)
  foreach(__list_select_current_arg ${${__list_select_lst}})
    __list_select_selector(${__list_select_current_arg})
    list(APPEND __res ${__ans})
  endforeach()
  return_ref(__res)  
endfunction()