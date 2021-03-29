
function(message_indent_pop)
  map_pop_back(global message_indent_level)
  ans(old_level)
  message_indent_level()
  ans(current_level)
  return_ref(current_level)
endfunction()
