## `(<path>)-><qualified path>`
##
## returns the fully qualified path name for path
## if path is a fully qualified name it returns path
## else path is interpreted as the relative path 
function(path path)
  pwd()
  ans(pwd)
  path_qualify_from("${pwd}" "${path}")
  return_ans()
endfunction()


