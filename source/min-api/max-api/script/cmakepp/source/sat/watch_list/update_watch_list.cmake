
  ## updates the watch list 
  ## removes newly assigned literal
  ## add watches to next unassigned literal 
  function(update_watch_list f watch_list assignments new_assignment)

    map_tryget("${watch_list}" ${new_assignment})
    ans(watched_clauses)

    map_remove("${watchlist}" ${new_assignment})

    map_tryget(${f} clause_literals)
    ans(clause_literals)

    foreach(watched_clause ${watched_clauses})
      update_watch_list_clause("${f}" "${watch_list}" "${assignments}" "${watched_clause}")
    endforeach()

  endfunction()