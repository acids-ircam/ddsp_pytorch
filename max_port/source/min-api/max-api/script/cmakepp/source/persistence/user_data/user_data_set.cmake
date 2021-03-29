## sets and persists data for the current user specified by identified by <id> 
## nav can be empty or a "." which will set the data at the root level
## else it can be a navigation expressions which (see map_navigate_set)
## e.g. user_data_set(common_directories cmakepp.base_dir /home/user/mydir)
## results in common_directories to contain
## {
##   cmakepp:{
##     base_dir:"/home/user/mydir"
##   }
## }
function(user_data_set id nav)
  set(args ${ARGN})

  if("${nav}" STREQUAL "." OR "${nav}_" STREQUAL "_")
    user_data_write("${id}" ${ARGN})
    return_ans()
  endif()
  user_data_read("${id}")
  ans(res)
  map_navigate_set("res.${nav}" ${ARGN})
  user_data_write("${id}" ${res})
  return_ans()
endfunction()

