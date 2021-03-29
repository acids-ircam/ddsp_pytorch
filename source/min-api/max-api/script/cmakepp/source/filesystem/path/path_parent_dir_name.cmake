## `(<path>)-><segment>` 
## 
## returns the name of the directory in which the specified file or folder resides
function(path_parent_dir_name)
  path("${ARGN}")
  ans(path)
  path_parent_dir("${path}")
  ans(path)
  path_component("${path}" --file-name-ext)
  return_ans()
endfunction()