# takes the passed list and returns only its unique elements
# see cmake's list(REMOVE_DUPLICATES)
function(list_unique __list_unique_lst)
  list(LENGTH ${__list_unique_lst} __len)
  if(${__len} GREATER 1)
	 list(REMOVE_DUPLICATES ${__list_unique_lst})
  endif()
	return_ref(${__list_unique_lst})
endfunction()
