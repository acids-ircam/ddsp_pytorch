## `()-> <string..>`
##
## returns a list of available generators on current system
function(cmake_generator_list)
  cmake_lean(--help)
  ans(help_text)
  list_pop_front(help_text)
  ans(error)
  if(error)
    message(FATAL_ERROR "could not execute cmake")
  endif()
  if("${help_text}" MATCHES "\nGenerators\n\n[^\n]*\n(.*)")
    set(generators_text "${CMAKE_MATCH_1}")
  endif()


  string(REGEX MATCHALL "(^|\n)  [^ \t][^=]*=" generators "${generators_text}")
  set(result)
  foreach(generator ${generators})
    if("${generator}" MATCHES "  ([^ ].*[^ \n])[ ]*=")
      set(generator "${CMAKE_MATCH_1}")
      list(APPEND result "${generator}")
    endif()

  endforeach()

  map_set(global cmake_generators "${result}")
  function(cmake_generators)
    map_tryget(global cmake_generators)
    return_ans()
  endfunction()

  cmake_generators()
  return_ans()
endfunction()