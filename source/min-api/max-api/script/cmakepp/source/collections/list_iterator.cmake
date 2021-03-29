
## instanciates a list_iterator from the specified list
  function(list_iterator __list_ref)
    list(LENGTH ${__list_ref} __list_ref_len)
    return(${__list_ref} ${__list_ref_len} 0-1)
  endfunction()
