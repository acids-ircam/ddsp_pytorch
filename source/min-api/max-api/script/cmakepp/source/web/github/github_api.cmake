## github_api()
## 
## 
function(github_api)
  set(github_api_token)
  if(NOT "$ENV{GITHUB_DEVEL_TOKEN_ID}_" STREQUAL "_" )
    set(github_api_token "?client_id=$ENV{GITHUB_DEVEL_TOKEN_ID}&client_secret=$ENV{GITHUB_DEVEL_TOKEN_SECRET}")
  endif()
  set(api_uri "https://api.github.com")
  define_http_resource(github_api "${api_uri}/:path${github_api_token}")

  github_api(${ARGN})
  return_ans()
endfunction()