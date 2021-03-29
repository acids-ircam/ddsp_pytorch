
  function(encoded_list_pop_back __lst)
    list_pop_back(${__lst})
    ans(back)
    set(${__lst} ${${__lst}} PARENT_SCOPE)
    string_decode_list("${back}")
    return_ans()
  endfunction()
