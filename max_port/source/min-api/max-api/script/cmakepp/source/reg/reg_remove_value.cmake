
  ## removes the specified value from the registry value
function(reg_remove_value key value_name)
  reg_read_value("${key}" "${value_name}")
  ans(values)

  list_remove(values ${ARGN})
  reg_write_value("${key}" "${value_name}" "${values}")

  return()

endfunction()