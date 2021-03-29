## bitbucket_api()
## 
## 
function(bitbucket_api)

  set(bitbucket_api_token)
 # if(NOT "$ENV{BITBUCKET_API_TOKEN}_" STREQUAL "_" )
 #   set(bitbucket_api_token "?client_id=$ENV{BITBUCKET_API_TOKEN}&client_secret=$ENV{GITHUB_DEVEL_TOKEN_SECRET}")
#  endif()
  set(api_uri "https://api.bitbucket.org/2.0")
  define_http_resource(bitbucket_api "${api_uri}/:path${bitbucket_api_token}")

  bitbucket_api(${ARGN})
  return_ans()
endfunction()