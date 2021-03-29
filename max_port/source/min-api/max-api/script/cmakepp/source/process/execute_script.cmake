## `(<cmake code> [--pure] <args...>)-><execute result>`
##
## equivalent to `execute(...)->...` runs the specified code using `cmake -P`.  
## prepends the current `cmakepp.cmake` to the script  (this default behaviour can be stopped by adding `--pure`)
##
## all not specified `args` are forwarded to `execute`
##
function(execute_script script)
  set(args ${ARGN})

  list_extract_flag(args --no-cmakepp)
  ans(nocmakepp)

  if(NOT nocmakepp)
    cmakepp_config(cmakepp_path)
    ans(cmakepp_path)
    set(script "include(\"${cmakepp_path}\")\n${script}")
  endif()
  fwrite_temp("${script}" ".cmake")
  ans(script_file)
  ## execute add callback to delete temporary file
  execute("${CMAKE_COMMAND}" -P "${script_file}"  --on-terminated-callback "[]() rm(${script_file})" ${args}) 
  return_ans()
endfunction()

