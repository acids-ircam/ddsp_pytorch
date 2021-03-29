macro(arguments_anonymous_function __args_start __args_end)
  arguments_cmake_code(${__args_start} ${__args_end})
  anonymous_function_new("${__ans}")
endmacro()



