
function(process_handle_get pid)
  map_tryget(__process_handles ${pid})
  return_ans()
endfunction()