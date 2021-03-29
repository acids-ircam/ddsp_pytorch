

  function(encoded_list_pop_front __lst)
    list_pop_front(${__lst})
    ans(front)
    set(${__lst} ${${__lst}} PARENT_SCOPE)
    string_decode_list("${front}")
    return_ans()
  endfunction()

