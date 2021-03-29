# decodes semicolons in a string
  function(string_decode_semicolon str)
    string(ASCII  31 semicolon_code)
    string(REPLACE "${semicolon_code}" ";" str "${str}")
    return_ref(str)
  endfunction()



## faster
  function(string_decode_semicolon str)
    string_codes()
    eval("
      function(string_decode_semicolon str)
        string(REPLACE  \"${semicolon_code}\" \";\" str \"\${str}\" )
        set(__ans \"\${str}\" PARENT_SCOPE)
      endfunction()
    ")
    string_decode_semicolon("${str}")
    return_ans()
  endfunction()
