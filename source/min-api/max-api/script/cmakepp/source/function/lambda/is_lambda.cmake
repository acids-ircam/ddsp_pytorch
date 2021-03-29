
function(is_lambda callable)
  if("${callable}" MATCHES "^\\[[a-zA-Z0-9_ ]*]*\\]\\([[a-zA-Z0-9_ ]*\\)")
    return(true)
  endif()
    return(false)
endfunction()