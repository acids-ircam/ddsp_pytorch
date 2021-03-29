
# encodes parentheses in a string
  function(string_encode_parentheses str)
    string_codes()
    string(REPLACE "\(" "${paren_open_code}" str "${str}")
    string(REPLACE "\)" "${paren_close_code}" str "${str}")
    return_ref(str)
  endfunction()


