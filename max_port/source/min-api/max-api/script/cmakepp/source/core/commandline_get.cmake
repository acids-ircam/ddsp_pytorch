# returns the list of command line arguments
function(commandline_get)
  set(args)
  foreach(i RANGE ${CMAKE_ARGC})  
    set(current ${CMAKE_ARGV${i}})
    string(REPLACE \\ / current "${current}")
    list(APPEND args "${current}")    
  endforeach()  

  return_ref(args)
endfunction() 


## 
##
## returns script | configure | build
function(cmake_mode)

endfunction()
