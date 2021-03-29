
  ## appends all specified values to registry value if they are not contained already
  function(reg_append_if_not_exists key value_name)
    reg_read_value("${key}" "${value_name}")
    ans(values)
    set(added_values)
    foreach(arg ${ARGN})
      list_contains(values "${arg}")
      ans(res)
      if(NOT res) 
        list(APPEND values "${arg}")
        list(APPEND added_values "${arg}")
      endif()
    endforeach()

    string_decode_semicolon("${values}")
    ans(values)
    reg_write_value("${key}" "${value_name}" "${values}")
    return_ref(added_values)
  endfunction()

