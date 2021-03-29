## prints the specified file to the console
function(fprint path)
  fread("${path}")
  ans(res)
  _message("${res}")
  return()
endfunction()


function(fprint_try path)
  path_qualify(path)
  if(EXISTS "${path}")
    fprint("${path}")
  endif()
  returN()
endfunction()