function(callable input)
  string(MD5  input_key "${input}" )
  get_propertY(callable GLOBAL PROPERTY "__global_callables.${input_key}")

  if(NOT callable)
    callable_new("${input}")
    ans(callable)

    checksum_string("${callable}")
    ans(callable_key)

    map_set_hidden(__global_callables "${input_key}" ${callable})
    map_set_hidden(__global_callables "${callable_key}" ${callable})

    map_get_special(${callable} callable_function)
    ans(function)

    map_set_hidden(__global_callable_functions "${input_key}" ${function})
    map_set_hidden(__global_callable_functions "${callable_key}" ${function})

  endif()
  set(__ans ${callable} PARENT_SCOPE)
endfunction()
