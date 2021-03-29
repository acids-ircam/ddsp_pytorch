
  function(listing_include listing)
    listing_compile("${listing}")
    eval("${__ans}")
    return_ans()
  endfunction()
