# reads the file specified and returns its content
function(fread path)
  path("${path}")
  ans(path)
  file(READ "${path}" res)
  return_ref(res)
endfunction()