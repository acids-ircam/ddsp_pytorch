

function(bitbucket_default_branch user repo)
  set(api_uri "https://bitbucket.org/api/1.0")
  set(query_uri "${api_uri}/repositories/${user}/${repo}/main-branch" )

  http_get("${query_uri}" --json --silent-fail)
  ans(response)

  if(NOT response)
    return()
  endif()

  map_tryget(${response} name)
  ans(res)
  return_ref(res)

endfunction()
