

  ## appends a value to the specified windows registry value
  function(reg_append_value key value_name)
    reg_read_value("${key}" "${value_name}")
    ans(data)
    set(data "${data};${ARGN}")
    reg_write_value("${key}" "${value_name}" "${data}")
    return_ref(data)
  endfunction()
