
    function(cmake_function_signatures code)
      regex_cmake()
      string(REGEX MATCHALL "${regex_cmake_function_begin}" functions "${code}")
      return_ref(functions)
    endfunction()

