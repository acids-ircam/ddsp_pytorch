
## map_append_unique 
## 
## appends values to the <map>.<prop> and ensures 
## that <map>.<prop> stays unique 
function(map_append_unique map prop)
  map_tryget("${map}" "${prop}")
  ans(vals)
  list(APPEND vals ${ARGN})
  list_remove_duplicates(vals)
  map_set("${map}" "${prop}" ${vals})
endfunction()
