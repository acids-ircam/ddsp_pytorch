
  function(encoded_list_peek_back __lst)
    list_peek_back(${__lst})
    ans(back)
    string_decode_list("${back}")
    return_ans()
  endfunction()