


  function(encoded_list_remove_item __lst)
    string_encode_list("${ARGN}")
    if(NOT ${__lst})
      return()
    endif()
    list(REMOVE_ITEM ${__lst} ${__ans})
    set(${__lst} ${${__lst}} PARENT_SCOPE)
    return()
  endfunction()
  