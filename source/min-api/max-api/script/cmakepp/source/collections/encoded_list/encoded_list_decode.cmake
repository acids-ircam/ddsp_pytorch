

## faster
function(encoded_list_decode str)
  string_codes()
  eval("
  function(encoded_list_decode str)
    if(\"\${str}_\" STREQUAL \"${empty_code}_\")
      return()
    endif()
    string(REPLACE \"${bracket_open_code}\" \"[\"  str \"\${str}\")
    string(REPLACE \"${bracket_close_code}\" \"]\"  str \"\${str}\")
    string(REPLACE \"${semicolon_code}\" \";\"  str \"\${str}\")
    set(__ans \"\${str}\" PARENT_SCOPE)
  endfunction()
  ")
  encoded_list_decode("${str}")
  return_ans()
endfunction()
