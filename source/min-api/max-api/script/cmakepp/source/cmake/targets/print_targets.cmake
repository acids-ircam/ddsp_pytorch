# prints the list of known targets 
function(print_targets)
  target_list()
  ans(res)
  foreach(target ${res})
    message("${target}")
  endforeach()

endfunction()


function(print_project_tree)
  map_tryget(global project_map)
  ans(pmap)

  json_print(${pmap})
  return()

endfunction()


function(print_target target_name)
  target_get_properties(${target_name})
  ans(res)
  json_print(${res})
endfunction()