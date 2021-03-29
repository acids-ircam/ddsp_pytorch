function(cmake_string_escape str)
  string(REPLACE "\\" "\\\\" str "${str}")
  string(REPLACE "\"" "\\\"" str "${str}")
  string(REPLACE "(" "\\(" str "${str}")
  string(REPLACE ")" "\\)" str "${str}")
  string(REPLACE "$" "\\$" str "${str}") 
  string(REPLACE "#" "\\#" str "${str}") 
  string(REPLACE "^" "\\^" str "${str}") 
  string(REPLACE "\t" "\\t" str "${str}")
  string(REPLACE ";" "\\;" str "${str}")
  string(REPLACE "\n" "\\n" str "${str}")
  string(REPLACE "\r" "\\r" str "${str}")
  
  #string(REPLACE "\0" "\\0" str "${str}") unnecessary because cmake does not support nullcahr in string
  string(REPLACE " " "\\ " str "${str}")
  return_ref(str)
endfunction()


function(cmake_string_escape2 str)
  if("${str}" MATCHES "[ \"\\(\\)#\\^\t\r\n\\\;]")
    ## encoded list encode cmake string...
    string(REPLACE "\\" "\\\\" str "${str}")
    string(REGEX REPLACE "([; \"\\(\\)#\\^])" "\\\\\\1" str "${str}")
    string(REPLACE "\t" "\\t" str "${str}")
    string(REPLACE "\n" "\\n" str "${str}")
    string(REPLACE "\r" "\\r" str "${str}")  
  endif()
  return_ref(str)
endfunction()


function(cmake_string_escape3 str)
  if("${str}" MATCHES "[ \"\\(\\)#\\^\t\r\n\\]")
    ## encoded list encode cmake string...
    string(REPLACE "\\" "\\\\" str "${str}")
    string(REGEX REPLACE "([ \"\\(\\)#\\^])" "\\\\\\1" str "${str}")
    string(REPLACE "\t" "\\t" str "${str}")
    string(REPLACE "\n" "\\n" str "${str}")
    string(REPLACE "\r" "\\r" str "${str}")  
  endif()
  return_ref(str)
endfunction()