## returns a temporary path in the specified directory
## if no directory is given the global temp_dir is used isntead
function(path_temp)
  set(args ${ARGN})

  if("${args}_" STREQUAL "_")
    cmakepp_config(temp_dir)
    ans(tmp_dir)
    set(args "${tmp_dir}")
  else()
    path("${args}")
    ans(args)
  endif()

  path_vary("${args}/mktemp")
  ans(path)

  return_ref(path)
endfunction()