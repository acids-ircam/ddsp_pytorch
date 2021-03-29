
  function(ascii_code char)
    generate_ascii_table()
    map_tryget(ascii_table "'${char}'")
    return_ans()
  endfunction()