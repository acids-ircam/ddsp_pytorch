## this file should not have the extension .cmake 
## because it needs to be included manually and last
## adds a callable as a task which is to be invoked later
function(task_enqueue callable)

  ## semicolon encode before string_encode_semicolon exists
  string(ASCII  31 us)
  string(REPLACE ";" "${us}" callable "${callable}")
  set_property(GLOBAL APPEND PROPERTY __initial_invoke_later_list "${callable}") 
  
  if(cmakepp_is_loaded)
    function(task_enqueue callable)
      task_new("${callable}")
      ans(task)
      task_queue_global()
      ans(task_queue)
      task_queue_push("${task_queue}" "${task}")
      return_ref(task)
    endfunction()
    address_get(__initial_invoke_later_list)
    ans(tasks)
    foreach(task ${tasks})
      string_decode_semicolon("${task}")
      ans(task)
      task_enqueue("${task}")
    endforeach()
  endif()
endfunction()

# initial version of task_enqueue which is used before cmakepp is loaded
# ## create invoke later functions 
# function(task_enqueue callable)
#   ## semicolon encode before string_encode_semicolon exists
#   string(ASCII  31 us)
#   string(REPLACE ";" "${us}" callable "${callable}")
#   set_property(GLOBAL APPEND PROPERTY __initial_invoke_later_list "${callable}") 
#   return()
# endfunction()
