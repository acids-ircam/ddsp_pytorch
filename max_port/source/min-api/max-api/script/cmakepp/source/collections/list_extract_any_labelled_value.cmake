## extracts any of the specified labelled values and returns as soon 
## the first labelled value is found
## lst contains its original elements without the labelled value 
function(list_extract_any_labelled_value __list_extract_any_labelled_value_lst)
  set(__list_extract_any_labelled_value_res)
  foreach(label ${ARGN})
    list_extract_labelled_value(${__list_extract_any_labelled_value_lst} ${label})
    ans(__list_extract_any_labelled_value_res)
    if(NOT "${__list_extract_any_labelled_value_res}_" STREQUAL "_")    
      break()
    endif()
  endforeach()
  set(${__list_extract_any_labelled_value_lst} ${${__list_extract_any_labelled_value_lst}}  PARENT_SCOPE)
  return_ref(__list_extract_any_labelled_value_res)
endfunction()
