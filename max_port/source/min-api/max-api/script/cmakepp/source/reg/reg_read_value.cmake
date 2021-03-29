
  ## reads the specified value from the windows registry  
  function(reg_read_value key value_name)
    reg_query_values("${key}")
    ans(res)
    map_tryget(${res} "${value_name}")
    ans(res)
    
    return_ref(res)
  endfunction()
