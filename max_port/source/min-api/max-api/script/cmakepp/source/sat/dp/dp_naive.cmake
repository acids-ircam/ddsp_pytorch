##
##
## naive implementation of Basic Davis-Putnam Backtrack Search
## http://www.princeton.edu/~chaff/publication/DAC2001v56.pdf
##
function(dp_naive f)
  dp_naive_init(${f})
  ans(context)
  set(initial_context ${context})
  while(true)
    ## decide which literal to try next to satisfy clauses
    ## returns true if decision was possible
    ## if no decision is made all clauses are satisfied
    ## and the algorithm terminates with success
    dp_naive_decide()
    ans(decision)
    if(NOT decision)
      dp_naive_finish(satsifiable)
      return_ans()
    endif()

    ## propagate decision 
    ## if a conflict occurs backtrack 
    ## when backtracking is impossible the algorithm terminates with failure
    while(true)

      dp_naive_bcp()
      ans(bcp)
      if(bcp)
        break()
      endif()

      ## backtrack 
      dp_naive_resolve_conflict()
      ans(resolved)

      if(NOT resolved)
        dp_naive_finish(not_satisfiable)
        return_ans()
      endif()
    endwhile()
  endwhile()

  message(FATAL_ERROR "unreachable code")
endfunction()


function(dp_naive_finish outcome)
  
  if("${outcome}" STREQUAL "satsifiable")
    map_peek_back(${context} decision_stack)
    ans(dl)
    map_tryget(${dl} assignments)
    ans(assignments)

    map_new()
    ans(result)
    map_set(${result} success true)
    map_set(${result} outcome ${outcome})
    map_set(${result} context ${context})
    map_set(${result} initial_context ${initial_context})
    map_set(${result} assignments ${assignments})
    return(${result})
  else()

    map_new()
    ans(result)
    map_set(${result} success false)
    map_set(${result} outcome ${outcome})
    map_set(${result} context ${context})
    map_set(${result} initial_context ${initial_context})
    #map_set(${result} assignments)
    return(${result})
  endif()
  return()
endfunction()


function(dp_naive_init f)
  map_import_properties(${f} clause_literal_map)
  ## add decision layer NULL to decision stack
  
  map_new()
  ans(assignments)
  map_duplicate(${clause_literal_map})
  ans(clauses)

  map_new()
  ans(decision_layer)
  map_set(${decision_layer} depth 0)
  map_set(${decision_layer} decision ${decision})
  map_set(${decision_layer} value false)
  map_set(${decision_layer} tried_both_ways false)
  map_set(${decision_layer} clauses ${clauses})
  map_set(${decision_layer} assignments "${assignments}")
  map_set(${decision_layer} parent)


  map_new()
  ans(context)

  map_set(${context} f ${f})
  map_set(${context} decision_stack ${decision_layer})

  return(${context})
endfunction()

function(dp_naive_push_decision parent decision value tried_both_ways)

  map_import_properties(${parent} clauses assignments)

  map_tryget(${context} decision_stack)
  ans(decision_stack)
  list(LENGTH decision_stack decision_depth)

  map_duplicate(${clauses})
  ans(clauses)

  map_duplicate(${assignments})
  ans(assignments)

  map_new()
  ans(dl)

  map_set(${dl} depth ${decision_depth})
  map_set(${dl} decision ${decision})
  map_set(${dl} value ${value})
  map_set(${dl} tried_both_ways ${tried_both_ways})
  map_set(${dl} clauses ${clauses})
  map_set(${dl} assignments ${assignments})
  map_set(${dl} parent ${parent})
  #message(PUSH FORMAT "decided {decision} (DL{dl.depth} {context.f.literal_map.${decision}}={value})")

  map_push_back(${context} decision_stack ${dl})
endfunction()

## return false if no unassigned variables remain
## true otherwise
## adds new decision layer to decision stack
function(dp_naive_decide)
  map_peek_back(${context} decision_stack)
  ans(dl)

  map_import_properties(${dl} clauses)
  map_values(${clauses})
  ans(unassigned_literals)


  list(LENGTH unassigned_literals unassigned_literals_count)
  if(NOT unassigned_literals_count)
    return(false)
  endif()

  list(GET unassigned_literals 0 decision)

  dp_naive_push_decision(${dl} ${decision} true false)
  return(true)
endfunction()

function(dp_naive_bcp)
  map_import_properties(${context} f)

  map_peek_back(${context} decision_stack)
  ans(dl)

  map_import_properties(${dl} decision value clauses assignments)
  map_set(${assignments} ${decision} ${value})
  
  #print_vars(clauses assignments)
  bcp("${f}" "${clauses}" "${assignments}" ${decision})
  ans(result)
  #print_vars(clauses assignments)

  #message(FORMAT "propagating {context.f.literal_map.${decision}} = ${value} => deduced: ${result}")
  #foreach(li ${result})
   # message(FORMAT "  {context.f.literal_map.${li}}=>{assignments.${li}}")
 # endforeach()
  if("${result}" MATCHES "(conflict)|(unsatisfied)")
    return(false)
  endif()

  return(true)
endfunction()



function(dp_naive_resolve_conflict)
  map_import_properties(${context} f)

  ## undo decisions until a decision is found which was not 
  ## tried the `other way` ie inversing the literals value
  set(conflicting_decision)
  while(true)
    map_pop_back(${context} decision_stack)
    ans(dl)
    ## store conflicting_decision
    map_set(${dl} conflicting_decision ${conflicting_decision})
    set(conflicting_decision ${dl})
    map_tryget(${dl} tried_both_ways)
    ans(tried_both_ways)
    if(NOT tried_both_ways)
      break()
    endif()
  endwhile()


  # d = most recent decision not tried `both ways`
  map_tryget(${dl} decision)
  ans(d)
  if("${d}_" STREQUAL "_")
    ## decision layer 0 reached -> cannot resolve
    return(false)
  endif()


  ## flip value
  map_tryget(${dl} value)
  ans(value)
  eval_truth(NOT value)
  ans(value)

  map_tryget(${dl} parent)
  ans(parent)

  ## pushback decision layer with value inverted
  dp_naive_push_decision(${parent} ${d} ${value} true)

  return(true)
endfunction()