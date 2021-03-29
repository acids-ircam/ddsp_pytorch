
  
## github_repository(<user> <repo>)-> {
##  full_name:
##  default_branch: 
## }
##
##  returns a github repository object if the repo exists
function(github_repository user repo)
  github_api("repos/${user}/${repo}" --silent-fail)
  ans(res)
  if(NOT res)
    return()
  endif()

  json_extract_string_value("default_branch" "${res}")
  ans(default_branch)
  json_extract_string_value("full_name" "${res}")
  ans(full_name)

  map_capture_new(full_name default_branch)
  return_ans()


endfunction()