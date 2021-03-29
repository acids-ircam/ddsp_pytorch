## `(<~task>)-><task>`
##
## coerces taskish to `<task>`
function(task taskish)
  map_get_special("${taskish}" $type)
  ans(type)
  if("${type}_" STREQUAL "task_" AND NOT ARGN)
    return("${taskish}")
  endif()
  task_new("${taskish}" ${ARGN})
  return_ans()
endfunction()