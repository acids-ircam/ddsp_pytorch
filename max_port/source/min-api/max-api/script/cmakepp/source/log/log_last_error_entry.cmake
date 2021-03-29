## `log_last_error_entry()-><log entry>`
##
## returns the last log entry which is an error
## 
function(log_last_error_entry)
  address_get(log_record)
  ans(log_record)
  set(entry)
  while(true)
    if(NOT log_record)
      break()
    endif()
    list_pop_back(log_record)
    ans(entry)

    map_tryget(${entry} type)
    ans(type)
    if(type STREQUAL "error")
      break()
    endif()
  endwhile()
  return_ref(entry)
endfunction()

