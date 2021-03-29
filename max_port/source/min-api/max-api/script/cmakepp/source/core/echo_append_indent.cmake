
  function(echo_append_indent)
    message_indent_get()
    ans(indent)

    echo_append("${indent} ${ARGN}")
    return()
  endfunction()