## `log_last_error_message()-><string>`
##
## returns the last logged error message
##
function(log_last_error_message)
  log_last_error_entry()
  ans(entry)
  if(NOT entry)
    return()
  endif()

  map_tryget(${entry} message)
  ans(message)


  return_ref(message)
endfunction()
