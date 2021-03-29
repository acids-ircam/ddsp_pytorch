# reads a single file from a git repository@branch using the 
# repository relative path ${path}. returns the contents of the file
function(git_read_single_file repository branch path )
  mktemp()
  ans(tmp_dir)

  set(branch_arg)
  if(branch)
    set(branch_arg --branch "${branch}") 
  endif()

  git_lean(clone --no-checkout ${branch_arg} --depth 1 "${repository}" "${tmp_dir}")
  ans_extract(error)

  if(error)
    rm(-r "${tmp_dir}")
    popd()
    return()
  endif()

  if(NOT branch)
    set(branch HEAD)
  endif()


  pushd("${tmp_dir}")
  git_lean(show --format=raw "${branch}:${path}")
  ans_extract(error)
  ans(res)

  popd()


  popd()
  rm(-r "${tmp_dir}")

  
  if(error)
    return()
  endif()
  

  return_ref(res)
  
endfunction()