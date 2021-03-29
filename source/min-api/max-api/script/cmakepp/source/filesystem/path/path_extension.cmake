## `(<path>)-><string>`
## 
## retuns the extension of the specified path
## 
function(path_extension path)
  path("${path}")
  ans(path)
  get_filename_component(res "${path}" EXT)
  return_ref(res)  
endfunction()