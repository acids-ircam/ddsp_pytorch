## `(<promise>)-> <promise state>`
##
## ```
## <promise state> ::= "resolved"|"pending"
## ```
## returns the state of the specified promise
macro(promise_state promise)
  map_tryget("${promise}" promise_state)
endmacro()