
  ## same as user_data_write except that an <obj> is parsed 
  function(user_data_write_obj id obj)
    obj("${obj}")
    ans(obj)
    user_data_write("${id}" "${obj}")
    return_ans()
  endfunction()

