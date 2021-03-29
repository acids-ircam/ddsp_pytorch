
 function(hg_constraint)
  map_has_all("${ARGN}" uri branch)
  ans(is_hg_constraint)
  if(is_hg_constraint)
    return("${ARGN}")
  endif() 

  package_query("${ARGN}")
  ans(pq)

  map_new()
  ans(constraint)
  nav(hg_constraint = pq.package_constraint)

  string_split_at_last(repo_uri branch "${hg_constraint}" "@")
  if(NOT branch)
    set(branch "default")
  endif()
  map_set(${constraint} uri "${repo_uri}")
  map_set(${constraint} "branch" ${branch})
  return (${constraint})
 endfunction()

 
function(line_info)
  set(t1 ${CMAKE_CURRENT_LIST_FILE})
  set(t2 ${CMAKE_CURRENT_LIST_LINE})
  obj("{
    file:$t1,
    line:$t2
    }")

  json_print(${__ans})
endfunction()