##
##
## tries to deserialize a file file.*  
function(fopen_data file)
  glob_path("${file}")
  ans(file)

  if(NOT EXISTS "${file}" OR IS_DIRECTORY "${file}")
    glob("${file}.*") 
    ans(file)
    list(LENGTH file len)
    if(NOT ${len} EQUAL 1)
      return()
    endif()
    if(IS_DIRECTORY "${file}")
      return()
    endif()
  endif()

  fread_data("${file}" ${ARGN})
  return_ans()
endfunction()