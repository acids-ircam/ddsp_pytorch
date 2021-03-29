# convenience function for accessing subversion
# use cd() to navigate to working directory
# usage is same as svn command line client
# syntax differs: svn arg1 arg2 ... -> svn(arg1 arg2 ...)
# also see wrap_executable for usage
# add a --process-handle flag to get a object containing return code, output
# input args etc.
# add --exit-code flag to get the return code of the commmand
# by default fails if return code is not 0 else returns  stdout/stderr
function(svn)
  find_package(Subversion)
  if(NOT SUBVERSION_FOUND)
    message(FATAL_ERROR "subversion is not installed")
  endif()
  # to prohibit non utf 8 decode errors
  set(ENV{LANG} C)
  set(ENV{LC_MESSAGES} C)
  
  wrap_executable(svn "${Subversion_SVN_EXECUTABLE}")
  
  svn(${ARGN})
  return_ans()
endfunction()