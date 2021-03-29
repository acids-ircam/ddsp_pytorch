## `([<path>])-><cmakelists>|<null>`
##
## opens a the closests cmakelists file (anchor file) found in current or parent directory
## returns nothing if no cmakelists file is found. 
function(cmakelists_open)
  file_find_anchor("CMakeLists.txt" ${ARGN})
  ans(cmakelists_path)
  if(NOT cmakelists_path)
    return()
  else()
    fread("${cmakelists_path}")
    ans(content)
  endif()
  cmakelists_new("${content}" "${cmakelists_path}")
  return_ans()
endfunction()