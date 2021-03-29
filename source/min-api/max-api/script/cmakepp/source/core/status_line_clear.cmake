function(status_line_clear)

  string_repeat(" " 100)
  ans(whitespace)

  eval("

    function(status_line_clear)
      map_tryget(global status)
      ans(status)
      if(\"\${status}_\" STREQUAL \"_\")
        return()
      endif()

      echo_append(\"\r${whitespace}\r\")
    endfunction()
  ")
  status_line_clear()
endfunction()

