  function(decode str)
    string(REPLACE "" ";"  str "${str}")
    string(REPLACE ""  "(" str "${str}")
    string(REPLACE ""  ")" str "${str}")
    string(REPLACE ""  "[" str "${str}")
    string(REPLACE ""  "]" str "${str}")
    set(__ans "${str}" PARENT_SCOPE)
  endfunction()