 ## `(<unquoated cmake code>)-><any>`
  ##
  ## evals the unquoted cmake code
  ## **example**
  ## ```
  ## set(callback message)
  ## eval_unquoted(
  ##   function(my_func)
  ##     ${callback}(hello \${ARGN} )
  ##     return(ok)
  ##   endfunction()
  ##   return(huhu)
  ## )
  ## ans(result) #-> result contains 'huhu'
  ## my_func(tobias) # -> returns 'ok' and prints 'hello tobias'
  ## ```
  function(eval_cmake)
    arguments_cmake_code(0 ${ARGC})
    ans(code)
    eval("${code}")
    return_ans()
  endfunction()