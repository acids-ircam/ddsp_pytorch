# retruns a portion of the list specified.
# negative indices count from back of list 
#
function(list_slice __list_slice_lst start_index end_index)
  # indices equal => select nothing

  list_normalize_index(${__list_slice_lst} ${start_index})
  ans(start_index)
  list_normalize_index(${__list_slice_lst} ${end_index})
  ans(end_index)

  if(${start_index} LESS 0)
    message(FATAL_ERROR "list_slice: invalid start_index ")
  endif()
  if(${end_index} LESS 0)
    message(FATAL_ERROR "list_slice: invalid end_index")
  endif()
  # copy array
  set(res)
  index_range(${start_index} ${end_index})
  ans(indices)

  list(LENGTH indices indices_len)
  if(indices_len)
    list(GET ${__list_slice_lst} ${indices} res)
  endif()
  #foreach(idx ${indices})
   # list(GET ${__list_slice_lst} ${idx} value)
    #list(APPEND res ${value})
   # message("getting value at ${idx} from ${${__list_slice_lst}} : ${value}")
  #endforeach()
 # message("${start_index} - ${end_index} : ${indices} : ${res}" )
  return_ref(res)
endfunction()


