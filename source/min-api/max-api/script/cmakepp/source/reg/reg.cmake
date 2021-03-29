

  ## access to the windows reg command
  function(reg)
    if(NOT WIN32)
      message(FATAL_ERROR "reg is not supported on non-windows platforms")
    endif()
    wrap_executable("reg" "REG")
    reg(${ARGN})
    return_ans()
  endfunction()

  function(reg_lean)

    if(NOT WIN32)
      message(FATAL_ERROR "reg is not supported on non-windows platforms")
    endif()
    wrap_executable_bare("reg_lean" "REG")
    reg_lean(${ARGN})
    return_ans()
  endfunction()