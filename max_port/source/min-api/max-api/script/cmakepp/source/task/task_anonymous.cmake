## `(<arguments: <any>...> <cmake code>...)-><task>`
##
## convenience functions which creates a task from the specified
## cmake code
## **example**
## ```
##  task_anonymous("1;2;3" (a b c) message("\${a} \${b} \${c}"))
##  ans(task)
##  task_invoke("${task}") ## prints '1 2 3'
## ```
function(task_anonymous arguments)
  arguments_cmake_code(1 ${ARGC})
  ans(code)
  anonymous_function_new("${code}")
  ans(func)
  task_new("${func}" ${arguments})
  return_ans()
endfunction()

