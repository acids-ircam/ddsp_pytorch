
## returns the git base dir (the directory in which .git is located)
function(git_base_dir)  
  git_dir("${ARGN}")
  ans(res)
  path_component("${res}" --parent-dir)
  ans(res)
  return_ref(res)
endfunction()