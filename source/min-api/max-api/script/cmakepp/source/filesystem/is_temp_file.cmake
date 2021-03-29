##
## 
## returns true if the specified file is a temp file (created by fwrite_temp)
function(is_temp_file file)
  if("${file}" MATCHES ".*\\/fwrite_temp[^ ]*")
    return(true)
  endif()
  return(false)
endfunction()