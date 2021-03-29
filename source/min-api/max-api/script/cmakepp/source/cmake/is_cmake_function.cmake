

    function(is_cmake_function code) 
      if("${code}" MATCHES "function.*endfunction")
        return(true)
      endif()
      return(false)
    endfunction()