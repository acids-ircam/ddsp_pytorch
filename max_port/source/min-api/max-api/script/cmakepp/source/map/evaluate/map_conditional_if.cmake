

  function(map_conditional_if parameters)
    
    set(map ${ARGN})

    map_tryget("${map}" $if)
    ans(condition)
  

        
    map_has("${map}" "$then")
    ans(has_then)

    map_has("${map}" "$else")
    ans(has_else)

    set(else)
    set(then)
    if(has_then OR has_else)
      map_tryget("${map}" $then)
      ans(then)
      map_tryget("${map}" $else)
      ans(else)
    else()
      set(then "${map}")
    endif()
    


    map_conditional_predicate_eval("${parameters}" "${condition}")    
    ans(evaluatedConditions)





    if(evaluatedConditions)
      set(result "${then}")
    else()
      set(result "${else}")
    endif()


    map_clone_shallow("${result}")
    ans(cloned)
    

    if("${result}_" STREQUAL "${map}_")
      map_keys_remove("${cloned}" $if $then $else)
    endif()
    


    map_conditional_evaluate("${parameters}" ${cloned})
    return_ans()
    
  endfunction()
