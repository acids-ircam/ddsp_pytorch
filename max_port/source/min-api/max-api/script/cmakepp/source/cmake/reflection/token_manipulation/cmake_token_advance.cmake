## `(<&<token>>)-><token>`
##
## advances the current token to the next token
macro(cmake_token_advance token_ref)
  map_tryget(${${token_ref}} next)
  ans(${token_ref})
endmacro()
