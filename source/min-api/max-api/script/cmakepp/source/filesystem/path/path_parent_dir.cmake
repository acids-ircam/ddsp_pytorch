## `(<path>)-><qualified path>` 
##
## returns the parent directory of the specified file or folder
## 
function(path_parent_dir path)
  path_qualify(path)
  get_filename_component(res "${path}" PATH)
  return_ref(res)
endfunction()