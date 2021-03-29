# decodes an encoded list
  function(string_decode_list str)
    string_decode_semicolon("${str}")
    ans(str)
    string_decode_bracket("${str}")
    ans(str)
    string_decode_empty("${str}")
    ans(str)
   # message("decoded3: ${str}")
    return_ref(str)
  endfunction()


## faster
function(string_decode_list str)
  string_codes()
  eval("
  function(string_decode_list str)
    string(REPLACE \"${bracket_open_code}\" \"[\"  str \"\${str}\")
    string(REPLACE \"${bracket_close_code}\" \"]\"  str \"\${str}\")
    string(REPLACE \"${semicolon_code}\" \";\"  str \"\${str}\")
    set(__ans \"\${str}\" PARENT_SCOPE)
  endfunction()
  ")
  string_decode_list("${str}")
  return_ans()
endfunction()