## **`delimiters()->[delimiter_begin, delimiter_end]`**
##
## parses delimiters and retruns a list of length 2 containing the specified delimiters. 
## The usefullness of this function becomes apparent when you use [string_take_delimited](#string_take_delimited)
## 
##
function(delimiters)
  set(delimiters ${ARGN})


  if("${delimiters}_" STREQUAL "_")
    set(delimiters \")
  endif()



  list_pop_front(delimiters)
  ans(delimiter_begin)


  if("${delimiter_begin}" MATCHES ..)
    string(REGEX REPLACE "(.)(.)" "\\2" delimiter_end "${delimiter_begin}")
    string(REGEX REPLACE "(.)(.)" "\\1" delimiter_begin "${delimiter_begin}")
  else()
    list_pop_front(delimiters)
    ans(delimiter_end)
  endif()

  
  if("${delimiter_end}_" STREQUAL "_")
    set(delimiter_end "${delimiter_begin}")
  endif()

  return(${delimiter_begin} ${delimiter_end})
endfunction()