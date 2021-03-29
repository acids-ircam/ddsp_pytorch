

  ## removes all duplicat values form the specified registry value
  function(reg_remove_duplicate_values key value_name)
    reg_read_value("${key}" "${value_name}")
    ans(values)
    list(REMOVE_DUPLICATES values)
    reg_write_value("${key}" "${value_name}" "${values}")
    return()
  endfunction()
