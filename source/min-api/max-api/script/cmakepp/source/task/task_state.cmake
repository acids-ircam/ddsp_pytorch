## `(<task>)-><task state>`
## 
## ```
## <task state> ::= "completed"|"running"
## ```
function(task_state task)
  map_has("${task}" return_value)
  ans(is_complete)
  if(is_complete)
    return("completed")
  endif()
  return("running")
endfunction()
