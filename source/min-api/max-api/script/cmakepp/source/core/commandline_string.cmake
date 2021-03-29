# returns the list of command line arguments
function(commandline_string)
  set(args)
  foreach(i RANGE ${CMAKE_ARGC})  
    set(current ${CMAKE_ARGV${i}})
    string(REPLACE \\ / current "${current}")
    set(args "${args} ${current}")
    
  endforeach()  

  return_ref(args)
endfunction() 
