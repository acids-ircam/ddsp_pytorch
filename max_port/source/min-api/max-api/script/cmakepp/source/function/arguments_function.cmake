
macro(arguments_function __function_name __args_start __args_end)
  arguments_cmake_code(${__args_start} ${__args_end})
  ans(code)
  string(REGEX REPLACE "^\\(" "(${__function_name}" code "${code}")
  eval("function${code}\nendfunction()")
endmacro()

