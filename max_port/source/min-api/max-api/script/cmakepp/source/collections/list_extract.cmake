# extracts elements from the list
# example
# set(lst 1 2  )
# list_extract(lst a b c)
# a contains 1
# b contains 2
# c contains nothing
# returns the rest of list
function(list_extract __list_extract_lst)
  set(__list_extract_list_tmp ${${__list_extract_lst}})
  set(args ${ARGN})
  while(true)
    list_pop_front( args)
    ans(current_arg)
    if(NOT current_arg)
      break()
    endif()
    list_pop_front( __list_extract_list_tmp)
    ans(current_value)
    set(${current_arg} ${current_value} PARENT_SCOPE)
  endwhile()
  return_ref(__list_extract_list_tmp)
endfunction()




