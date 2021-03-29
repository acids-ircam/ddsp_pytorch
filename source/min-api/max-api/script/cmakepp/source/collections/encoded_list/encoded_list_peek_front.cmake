
  function(encoded_list_peek_front __lst)
    list_peek_front(${__lst})
    ans(front)
    string_decode_list("${front}")
    return_ans()
  endfunction()
