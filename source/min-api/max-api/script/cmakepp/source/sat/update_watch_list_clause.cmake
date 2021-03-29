##
##
## updates a single clause int the watch list
function(update_watch_list_clause f watch_list assignments watched_clause)
  map_tryget("${f}" clause_literals)
  ans(clause_literals)

  map_tryget(${clause_literals} ${watched_clause})
  ans(watched_clause_literals)
  
  ## loop through all literals for watched clause
  ## get the currently watched literals from watch clause
  set(current_watch_count 0)

  while(${current_watch_count} LESS 2 AND NOT "${watched_clause_literals}_" STREQUAL "_" )
    list_pop_front(watched_clause_literals)
    ans(current_literal)
    if(NOT "${current_literal}" EQUAL "${new_assignment}")
      map_tryget("${assignments}" "${new_assignment}")
      ans(is_assigned)
      if(NOT is_assigned)
        map_append_unique("${watch_list}" "${current_literal}" "${watched_clause}")
        math(EXPR current_watch_count "${current_watch_count} + 1")
      endif()
    endif()
  endwhile()
endfunction()
