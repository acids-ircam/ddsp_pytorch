## `(<?path>)-> <qualified path...>`
##
## returns a list of files and folders in the specified directory
##
function(ls)
  path("${ARGN}")
  ans(path)

  if(IS_DIRECTORY "${path}")
    set(path "${path}/*")
  endif()

  file(GLOB files "${path}")
  return_ref(files)
endfunction()


