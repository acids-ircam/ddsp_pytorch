# turns the lazy cmake code into valid cmake
#
function(lazy_cmake cmake_code)
# normalize cmake 
  # 
  string(STRIP "${cmake_code}" cmake_code )
  if(NOT "${cmake_code}" MATCHES "[ ]*[a-zA-Z0-9_]+\\(.*\\)[ ]*")
    string(REGEX REPLACE "[ ]*([a-zA-Z0-9_]+)[ ]*(.*)" "\\1(\\2)" cmd "${cmake_code}")
    string(REGEX REPLACE "[ ]*([a-zA-Z0-9_]+)[ ]*(.*)" "\\1" cmdname "${cmake_code}")
    if(NOT COMMAND "${cmdname}")
      string(STRIP "${cmake_code}" cc)
      set(cmd "set_ans(\"\${${cc}}\")")
    endif()
  endif()



  return_ref(cmd)

endfunction()