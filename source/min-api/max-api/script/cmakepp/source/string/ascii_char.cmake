
  function(ascii_char code)
    ascii_generate_table()
    map_tryget(ascii_table "${code}")
    return_ans()
  endfunction()

 ## faster version
  function(ascii_char code)
    string(ASCII "${code}" res)
    return_ref(res)
  endfunction()