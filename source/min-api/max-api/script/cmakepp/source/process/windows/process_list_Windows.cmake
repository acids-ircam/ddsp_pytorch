
## platform specific implementation for process_list under windows
function(process_list_Windows)
  win32_wmic(process where "processid > 0" get processid) #ignore idle process
  ans(ids)


  string(REGEX MATCHALL "[0-9]+" matches "${ids}")
  set(ids)



  foreach(id ${matches})
    process_handle("${id}")
    ans(handle)
    list(APPEND ids ${handle})
  endforeach()



  return_ref(ids)
endfunction()