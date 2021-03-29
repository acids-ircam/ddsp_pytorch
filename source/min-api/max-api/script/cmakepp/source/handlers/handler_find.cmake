# returns those handlers in handler_lst which match the specified request  
  function(handler_find handler_lst request)
    set(result)
    foreach(handler ${${handler_lst}})
      handler_match(${handler} ${request})
      ans(res)
      if(res)
        list(APPEND result ${handler})
      endif()
    endforeach()

    return_ref(result)
  endfunction() 
