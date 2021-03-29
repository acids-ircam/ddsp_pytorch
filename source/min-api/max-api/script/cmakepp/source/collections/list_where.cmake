# executes a predicate on every item of the list (passed by reference)
# and returns those items for which the predicate holds
function(list_where __list_where_lst predicate)

	foreach(item ${${__list_where_lst}})
    rcall(__matched = "${predicate}"("${item}"))
		if(__matched)
			list(APPEND result_list ${item})
		endif()
	endforeach()
	return_ref(result_list)
endfunction()


## fast implemenation
function(list_where __list_where_lst __list_where_predicate)
  function_import("${__list_where_predicate}" as __list_where_predicate REDEFINE)
  set(__list_where_result_list)
  foreach(__list_where_item ${${__list_where_lst}})
    __list_where_predicate(${__list_where_item})
    ans(__matched)
    if(__matched)
      list(APPEND __list_where_result_list ${__list_where_item})
    endif()
  endforeach()
  return_ref(__list_where_result_list)
endfunction()
