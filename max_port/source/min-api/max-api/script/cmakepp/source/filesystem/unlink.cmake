## `(<path>)-><bool>`
## 
## unlinks the specified link without removing the links content.
function(unlink)
  wrap_platform_specific_function(unlink)
  unlink(${ARGN})
  return_ans()
endfunction()



function(unlink_Windows symlink)
  path_qualify(symlink)
  string(REPLACE "/" "\\" symlink "${symlink}") 
  win32_cmd_lean("/C" "rmdir" "${symlink}")
  ans_extract(res)
  if(res)
    return(false)
  endif()
  return(true)
endfunction()


function(unlink_Linux symlink)
  path_qualify(symlink)
  if("${symlink}" MATCHES "(.*)\\/$")
    set(symlink "${CMAKE_MATCH_1}")
  endif()
  execute_process(COMMAND "unlink" "${symlink}" RESULT_VARIABLE res)
  if(res)
    return(false)
  endif()

  return(true)
endfunction()