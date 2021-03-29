## github_repositories() -> {
##   full_name:
##   default_branch:
## }
##
## returns the list of repositories for the specified user
function(github_repository_list user)
  set(repositories)
    github_api("users/${user}/repos" --response)
    ans(res)
    assign(error = res.client_status)
    if(error)
      return()
    endif()
    assign(content = res.content)

    
    ## this is a quick way to get all full_name fields of the unparsed json
    ## parsing large json files would be much too slow
    json_extract_string_value(full_name "${content}")
    ans(full_names)
    json_extract_string_value(default_branch "${content}")
    ans(default_branches)

    set(repos)
    foreach(full_name ${full_names})
      list_pop_front(default_branches)
      ans(default_branch)
      map_capture_new(full_name default_branch)
      ans_append(repos)
    endforeach() 
    return_ref(repos)
endfunction()