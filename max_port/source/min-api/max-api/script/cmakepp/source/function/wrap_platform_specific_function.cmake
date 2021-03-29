## defines the function called ${function_name} to call an operating system specific function
## uses ${CMAKE_SYSTEM_NAME} to look for a function called ${function_name}${CMAKE_SYSTEM_NAME}
## if it exists it is wrapped itno ${function_name}
## else ${function_name} is defined to throw an error if it is called
function(wrap_platform_specific_function function_name)
  os()
  ans(os_name)
  set(specificname "${function_name}_${os_name}")
  if(NOT COMMAND "${specificname}")      
    eval("
    function(${function_name})
      message(FATAL_ERROR \"operation is not supported on ${os_name} - look at document of '${function_name}' and implement a function with a matching interface called '${specificname}' for you own system\")        
    endfunction()      
    ")
  else()
    eval("
      function(${function_name})
        ${function_name}_${os_name}(\${ARGN})
        return_ans()
      endfunction()
    ")
    
  endif()
  return()
endfunction()
