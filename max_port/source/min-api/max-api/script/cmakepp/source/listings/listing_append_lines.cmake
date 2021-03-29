

  function(listing_append_lines listing)
   foreach(line ${ARGN})
    listing_append(${listing} "${line}")
   endforeach()
  endfunction()