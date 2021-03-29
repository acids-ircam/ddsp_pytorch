## creates encoded lists from the specified arguments
function(encoded_list)
  arguments_encoded_list(0 ${ARGC})
  set(__ans "${__ans}" PARENT_SCOPE)
endfunction()


