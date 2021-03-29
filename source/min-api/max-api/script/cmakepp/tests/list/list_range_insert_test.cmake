function(test)


  function(list_range_insert lst_ref range)
    range_parse("${range}")
    ans(range)
    list(LENGTH range len)
    message("input_list '${${lst_ref}}'")

    set(removed)
    list_range_remove(${lst_ref} "${range}")
 

    message("range '${range}'")
    message("removed ${removed}")
    message("result_list '${${lst_ref}}'\n")

    return_ref(removed)
  endfunction()

  set(lstA a b c)
  list_range_insert(lstA "")
  ans(res)


  set(lstA)
  list_range_insert(lstA "" 1)
  ans(res)




  return_ref(removed)
  




endfunction()