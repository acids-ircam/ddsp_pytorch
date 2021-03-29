# encodes a string list so that it can be correctly stored and retrieved
function(string_encode_list str)
  string_codes()
  string(REPLACE "[" "${bracket_open_code}" str "${str}")
  string(REPLACE "]" "${bracket_close_code}" str "${str}")
  string(REPLACE ";" "${semicolon_code}" str "${str}")
  set(__ans "${str}" PARENT_SCOPE)
endfunction()

## faster
function(string_encode_list str)
  string_codes()
  eval("
    function(string_encode_list str)
    string(REPLACE \"[\" \"${bracket_open_code}\" str \"\${str}\")
    string(REPLACE \"]\" \"${bracket_close_code}\" str \"\${str}\")
    string(REPLACE \";\" \"${semicolon_code}\" str \"\${str}\")
    set(__ans \"\${str}\" PARENT_SCOPE)
  endfunction()
  ")
  string_encode_list("${str}")
  return_ans()
endfunction()


