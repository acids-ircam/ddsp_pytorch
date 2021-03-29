function(git_repository_name repository_uri)
  get_filename_component(repo_name "${repository_uri}" NAME_WE)
  return("${repo_name}")
endfunction()