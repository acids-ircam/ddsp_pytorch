## `(<&cmake token>)-><cmake token>`
## 
## the token ref contains the previous token after invocation
macro(cmake_token_go_back token_ref)
  map_tryget(${${token_ref}} previous)
  ans(${token_ref})
endmacro()

