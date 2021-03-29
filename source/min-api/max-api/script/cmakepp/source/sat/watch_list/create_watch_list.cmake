

function(create_watch_list f assignments)   
  map_new()
  ans(watch_list)
  
  map_tryget(${f} c_last)
  ans(c_last)
    
  foreach(ci RANGE 0 ${c_last})
    update_watch_list_clause("${f}" "${watch_list}" "${assignments}" "${ci}")
  endforeach()

  return_ref(watch_list)
endfunction()
