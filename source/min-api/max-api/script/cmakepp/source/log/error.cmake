## `error(...)-><log entry>`
##
## Shorthand function for `log(<message> <refs...> --error)
## 
## see [log](#log)
##
function(error)
  log(--error ${ARGN})  
  return_ans()
endfunction()


