# encodes brackets
function(string_encode_bracket str)
  string_codes()
  string(REPLACE "[" "${bracket_open_code}" str "${str}") 
  string(REPLACE "]" "${bracket_close_code}" str "${str}")
  return_ref(str)
endfunction()


## faster
function(string_encode_bracket str)
  string_codes()
  eval("
    function(string_encode_bracket str)
    string(REPLACE \"[\" \"${bracket_open_code}\" str \"\${str}\")
    string(REPLACE \"]\" \"${bracket_close_code}\" str \"\${str}\")
    set(__ans \"\${str}\" PARENT_SCOPE)
  endfunction()
  ")
  string_encode_bracket("${str}")
  return_ans()
endfunction()


