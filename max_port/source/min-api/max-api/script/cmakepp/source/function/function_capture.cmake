## captures variables from the current scope in the function
function(function_capture callable)
  set(args ${ARGN})
  list_extract_labelled_value(args as)
  ans(func_name)
  if(func_name STREQUAL "")
    function_new()
    ans(func_name)
  endif()

  set(captured_var_string)
  foreach(arg ${args})
    set(captured_var_string "${captured_var_string}set(${arg} \"${${arg}}\")\n")
  endforeach()

  function_import("${callable}")
  ans(callable)

  eval("
    function(${func_name})
      ${captured_var_string}
      ${callable}(\${ARGN})
      return_ans()
    endfunction()
  ")
  return_ref(func_name)
endfunction()


