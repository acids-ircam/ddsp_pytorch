


  macro(encoded_list_get __lst idx)
    list(GET ${__lst} ${idx} __ans)
    string_decode_list("${__ans}")
  endmacro()
