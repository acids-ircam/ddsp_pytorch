
  function(is_callable callable)
    map_get_special("${callable}" callable_function)
    ans(func)
    if(COMMAND "${func}")
      return(true)
    endif()
    return(false)
  endfunction()
