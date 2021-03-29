## `(<cmakepp code file>)-><cmake code file>`
##
## compiles the specified source file to enable expressions
## the target file can be specified. by default a temporary file is created
## todo:  cache result
function(cmakepp_compile_file source)
  set(target ${ARGN})

  fread("${source}")
  ans(content)
  cmakepp_expr_compile("${content}")
  ans(content)
  if(NOT target)
    fwrite_temp("${content}" cmakepp)
    ans(target)
  else()
    fwrite("${target}" "${content}")
    ans(target)
  endif()
  return_ref(target)
endfunction()
