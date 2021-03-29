  function(parse_string rstring definition_id)
    # initialize
    if(NOT __parse_string_initialized)
      set(args ${ARGN})
      set(__parse_string_initialized true)
      list_extract(args definitions parsers language)
      function_import_table(${parsers} __call_string_parser)
    endif()

    # 
    map_get("${definitions}" "${definition_id}")
    ans(definition)
    
    #
    map_get("${definition}" parser)
    ans(parser_id)
    
    #
  #  message(FORMAT "${parser_id} parser parsing ${definition_id}..")
    message_indent_push()
    __call_string_parser("${parser_id}" "${rstring}")
    ans(res)
    message_indent_pop()
   # message(FORMAT "${parser_id} parser returned: ${res} rest is")
   #list(LENGTH res len)
 #  if(len)
   #  message("parsed '${res}' with ${parser_id} parser")
   #endif()   
    return_ref(res)
  endfunction()