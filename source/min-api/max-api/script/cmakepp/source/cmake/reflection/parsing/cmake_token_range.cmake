## `(<cmake token range>|<cmake token>...|<cmake code>)-><cmake token range>`
##
## coerces the input to become a token range 
## if the input already is a token range it is returned
## if the input is a list of tokens the token range will be extracted
## if the input is a string it is assumed to be cmake code and parsed to return a token range
function(cmake_token_range )    
  cmake_tokens("${ARGN}")
  rethrow()
  ans(range)
  list_pop_front(range)
  ans(begin)
  list_pop_back(range)
  ans(end)
  return(${begin} ${end})
endfunction()
