
  function(token_stream_isempty stream)
    map_tryget(${stream}  current)
    ans(current)
    if(current)
      return(false)
    endif()
    return(true)

  endfunction()