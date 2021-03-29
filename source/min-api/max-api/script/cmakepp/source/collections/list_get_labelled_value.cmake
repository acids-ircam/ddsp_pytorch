## gets the labelled value from the specified list
## set(thelist a b c d)
## list_get_labelled_value(thelist b) -> c
function(list_get_labelled_value __list_get_labelled_value_lst __list_get_labelled_value_value)
  list_extract_labelled_value(${__list_get_labelled_value_lst} ${__list_get_labelled_value_value} ${ARGN})
  return_ans()
endfunction()