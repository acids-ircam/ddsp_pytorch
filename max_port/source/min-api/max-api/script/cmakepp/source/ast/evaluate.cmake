
  function(evaluate str language expr)
    language(${language})
    ans(language)

    set(scope ${ARGN})
    is_map("${scope}" )
    ans(ismap)
    if(NOT ismap)
      map_new()
      ans(scope)
      foreach(arg ${ARGN})
        map_set(${scope} "${arg}" ${${arg}})
      endforeach()
    endif()


    map_new()
    ans(context)
    map_set(${context} scope ${scope})

  #  message("expr ${expr}")

    ast("${str}" ${language} "${expr}")
    #return("gna")
    ans(ast) 
   # address_print(${ast})
    ast_eval(${ast} ${context} ${language})
    ans(res)
    if(NOT ismap)
      map_promote(${scope})
    endif()
    return_ref(res)
  endfunction()