
function(value_range_get value range)
  list(LENGTH value length)
  if("${range}" MATCHES "^(0|([1-9][0-9]*))$")
    if(NOT length)
      throw("index out of range: ${range}")
    endif()
    list(GET value "${range}" result)
    return_ref(result)
  endif()

  list_range_get(value "${range}")
  return_ans()
endfunction()

