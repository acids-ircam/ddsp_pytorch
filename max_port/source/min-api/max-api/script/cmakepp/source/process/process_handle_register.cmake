
function(process_handle_register process_handle)
  event_emit(on_process_handle_created ${process_handle})
endfunction()
