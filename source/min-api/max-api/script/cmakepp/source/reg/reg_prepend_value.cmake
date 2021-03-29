
  ## prepends a value to the specified windows registry value
  function(reg_prepend_value key value_name)
    reg_read_value("${key}" "${value_name}")
    ans(data)
    set(data "${ARGN};${data}")
    reg_write_value("${key}" "${value_name}" "${data}")
    return_ref(data)
  endfunction()
