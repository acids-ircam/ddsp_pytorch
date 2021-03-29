



  function(listing_make_compile)
    listing()
    ans(uut)
    foreach(line ${ARGN})
      listing_append(${uut} "${line}")
    endforeach()
    listing_compile(${uut})
    return_ans()
  endfunction()
