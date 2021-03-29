# searchs for label in lst. if label is found 
# the label and its following value is removed
# and returned
# if label is found but no value follows ${ARGN} is returned
# if following value is enclosed in [] the brackets are removed
# this allows mulitple values to be returned ie
# list_extract_labelled_value(lstA --test1)
# if lstA is a;b;c;--test1;[1;3;4];d
# the function returns 1;3;4
function(list_extract_labelled_value lst label)
  # return nothing if lst is empty
  list_length(${lst})
  ans(len)
  if(NOT len)
    return()
  endif()
  # find label in list
  list_find(${lst} "${label}")
  ans(pos)
  
  if("${pos}" LESS 0)
    return()
  endif()

  eval_math("${pos} + 2")
  ans(end)


  if(${end} GREATER ${len} )
    eval_math("${pos} + 1")
    ans(end)
  endif()

  list_erase_slice(${lst} ${pos} ${end})
  ans(vals)

  list_pop_front(vals)
  ans(flag)
    

  # special treatment for [] values
  if("_${vals}" MATCHES "^_\\[.*\\]$")
    string_slice("${vals}" 1 -2)
    ans(vals)
  endif()


  if("${vals}_" STREQUAL "_")
    set(vals ${ARGN})
  endif()

  
  set(${lst} ${${lst}} PARENT_SCOPE)


  return_ref(vals)
endfunction()
