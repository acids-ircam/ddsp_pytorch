
  ### returns true if the registry value contains the specified value
  function(reg_contains_value key value_name value)
    reg_read_value("${key}" "${value_name}")
    ans(values)
    list_contains(values "${value}")
    return_ans()
  endfunction()

