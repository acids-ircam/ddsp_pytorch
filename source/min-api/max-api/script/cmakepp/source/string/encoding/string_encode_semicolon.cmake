# encodes semicolons with seldomly used utf8 chars.
# causes error for string(SUBSTRING) command
  function(string_encode_semicolon str)
    # make faster by checking if semicolon exists?
    string(ASCII  31 semicolon_code)
    # string(FIND "${semicolon_code}" has_semicolon)
    #if(has_semicolon GREATER -1) replace ...

    string(REPLACE ";" "${semicolon_code}" str "${str}" )
    return_ref(str)
  endfunction()


## faster
  function(string_encode_semicolon str)
    string_codes()
    eval("
      function(string_encode_semicolon str)
        string(REPLACE \";\" \"${semicolon_code}\" str \"\${str}\" )
        set(__ans \"\${str}\" PARENT_SCOPE)
      endfunction()
    ")
    string_encode_semicolon("${str}")
    return_ans()
  endfunction()

