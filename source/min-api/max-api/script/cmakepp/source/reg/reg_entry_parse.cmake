

## parses the result of reg(query )call
## returns an reg entry object:
## {
##   "value_name":registry value name
##   "key":registry key
##   "value":value of the entry if it exists
##   "type": registry value type (ie REG_SZ) or KEY if its a key
## }
function(reg_entry_parse query line)
    if("${line}" MATCHES "^    ([^ ]+)")
      set(regex "^    ([^ ]+)    ([^ ]+)    (.*)")
      string(REGEX REPLACE "${regex}" "\\1" value_name "${line}")
      string(REGEX REPLACE "${regex}" "\\2" type "${line}")
      string(REGEX REPLACE "${regex}" "\\3" value "${line}")
      string_decode_semicolon("${value}")
      ans(value)
      
    else()
 # _message("line ${line}")
      set(key "${line}")
      set(type "KEY")
      set(value "")
      set(value_name "")
    endif()
    map_capture_new(key value_name type value)
    return_ans()
endfunction()
