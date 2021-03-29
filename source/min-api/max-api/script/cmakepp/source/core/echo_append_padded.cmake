
  function(echo_append_padded len str)
    string_pad("${str}" "${len}" " ")
    ans(str)
    echo_append("${str}")
  endfunction()