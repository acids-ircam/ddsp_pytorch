# sleeps for the specified amount of seconds
function(sleep seconds)
  if("${CMAKE_MAJOR_VERSION}" LESS 3)
    if(UNIX)
      execute_process(COMMAND sleep ${seconds} RESULT_VARIABLE res)

      if(NOT "${res}" EQUAL 0)
        message(FATAL_ERROR "sleep failed")
      endif()
      return()
    endif()

    message(WARNING "sleep no available in cmake version ${CMAKE_VERSION}")
    return()
  endif()

  cmake_lean(-E sleep "${seconds}")
  return()
endfunction()