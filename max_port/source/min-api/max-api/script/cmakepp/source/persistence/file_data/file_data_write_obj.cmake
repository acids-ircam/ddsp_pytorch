
  ## same as file_data_write except that an <obj> is parsed 
  function(file_data_write_obj dir id obj)
    obj("${obj}")
    ans(obj)
    file_data_write("${dir}" "${id}" "${obj}")
    return_ans()
  endfunction()
