
  function(listing_append listing line)
    string_combine(" " ${ARGN})
    ans(rest)
    string_encode_semicolon("${line}${rest}")
    ans(line)
    address_append("${listing}" "${line}")
    return()
  endfunction()