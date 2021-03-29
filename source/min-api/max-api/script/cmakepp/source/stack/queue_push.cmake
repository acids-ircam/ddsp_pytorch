
  function(queue_push queue)
    map_tryget("${queue}" back)
    ans(back)
    map_set_hidden("${queue}" "${back}" "${ARGN}")
    math(EXPR back "${back} + 1")
    map_set_hidden("${queue}" back "${back}")
    
  endfunction()