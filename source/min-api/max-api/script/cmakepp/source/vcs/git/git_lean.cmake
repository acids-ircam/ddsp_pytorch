## `(<args...>)`[<exitcode>, <stdout>]
##
## a lean wrapper for git
## does not take part in the process management of cmakepp
function(git_lean)
  find_package(Git)
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "missing git")
  endif()

  wrap_executable_bare(git_lean "${GIT_EXECUTABLE}")
  git_lean(${ARGN})
  return_ans()


endfunction()