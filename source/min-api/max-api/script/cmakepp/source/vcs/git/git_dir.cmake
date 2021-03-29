
# returns the git directory for pwd or specified path
function(git_dir)
  set(path ${ARGN})
  path("${path}")
  ans(path)
  message("${path}")

  pushd("${path}")
  git(rev-parse --show-toplevel)
  ans(res)
  message("${res}")
  
  popd()
  string(STRIP "${res}" res)
  set(res "${res}/.git")
  message("${res}")
  return_ref(res)
endfunction()
