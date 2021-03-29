function(test)

  find_package(Cygwin)

  if(NOT CYGWIN_INSTALL_PATH)
    message("test inconcolusive - cywgin is not available on this system")
    return()
  endif()

  message("test inconclusive")
  return()

  message("cygwin_executable ${CYGWIN_INSTALL_PATH}")

  function(cygwin)
    find_package(Cygwin)
    if(NOT CYGWIN_INSTALL_PATH)
      message(FATAL_ERROR "cygwin not available on this system")
    endif()

    wrap_executable(cygwin "${CYGWIN_INSTALL_PATH}/cywgin.bat")
    cygwin(${ARGN})
    return_ans()
  endfunction()

  cygwin()
  ans(res)
  message("${res}")

endfunction()