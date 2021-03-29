# parses a table as is output by win32 commands like tasklist
# the format is
# header1 header2 header3
# ======= ======= =======
# val1    val2    val3
# val4    val5    val6
# not that the = below the header is used as the column width and must be the max length of any value in 
# column including the header
# returns a list of <row> where row is a map and the headers are the keys   (values are trimmed from whitespace)
# the example above results in 
# {
#   "header1":"val1",
#   "header2":"val2",
#   "header3":"val3"
# }
#
function(table_deserialize input)
  string_lines("${input}")
  ans(lines)
  list_pop_front(lines)
  ans(firstline)  
  list_pop_front(lines)    
  ans(secondline)
  list_pop_front(lines)    
  ans(thirdline)

  string(REPLACE "=" "." line_match "${thirdline}")
  string_split("${line_match}" " ")
  ans(parts)
  list(LENGTH parts cols) 
  set(linematch)
  set(first true)
  foreach(part ${parts})
    if(first)
      set(first false)
    else()
      set(linematch "${linematch} ")
    endif()
    set(linematch "${linematch}(${part})")
  endforeach()

  set(headers __empty) ## empty is there to buffer so that headers can be index 1 based instead of 0 based
  foreach(idx RANGE 1 ${cols})
    string(REGEX REPLACE "${linematch}" "\\${idx}" header "${secondline}")
    string(STRIP "${header}" header)
    list(APPEND headers ${header})
  endforeach()



  set(result)
  foreach(line ${lines})
    map_new()
    ans(l)
    foreach(idx RANGE 1 ${cols})
      string(REGEX REPLACE "${linematch}" "\\${idx}" col "${line}")
      string(STRIP "${col}" col)
      list_get(headers ${idx})
      ans(header)
      map_set(${l} "${header}" "${col}")        
    endforeach()
    list(APPEND result ${i})
  endforeach()

  return_ref(result)
endfunction()
