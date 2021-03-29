## `(<cmakelists>)-> <bool>`
##
## closes the specified cmakelists file.  This causes it to be written to its path
## returns true on success
function(cmakelists_close cmakelists) 
  map_tryget(${cmakelists} path)
  ans(cmakelists_path)
  cmakelists_serialize("${cmakelists}")
  ans(content)
  fwrite("${cmakelists_path}" "${content}")
  return(true)
endfunction()