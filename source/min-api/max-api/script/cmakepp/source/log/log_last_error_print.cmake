## `log_last_error_print()-><void>`
##
## prints the last error message to the console  
##
function(log_last_error_print)
  log_last_error_entry()
  ans(entry)
  if(NOT entry)
    return()
  endif()

  message(FORMAT "Error in {entry.function}: {entry.message}")
  while(true)
    map_tryget(${entry} preceeding_error)
    ans(entry)
    if(NOT entry)
      break()
    endif()
    message(FORMAT "  because of {entry.function}: {entry.message}")
  endwhile()
  return()
endfunction()
