
## returns the regex for a delimited string 
## allows escaping delimiter with '\' backslash
function(regex_delimited_string)
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
  #set(regex "${delimiter_begin}(([^${delimiter_end}])*)${delimiter_end}")
  set(delimiter_end "${delimiter_end}" PARENT_SCOPE)
  #set(regex "${delimiter_begin}(([^${delimiter_end}\\]|(\\[${delimiter_end}])|\\\\)*)${delimiter_end}")
  regex_escaped_string("${delimiter_begin}" "${delimiter_end}")
  ans(regex)
  return_ref(regex)
endfunction()
