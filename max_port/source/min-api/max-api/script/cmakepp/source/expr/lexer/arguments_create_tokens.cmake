macro(arguments_create_tokens __start_idx __end_idx)
  arguments_tokenize("${__start_idx}" "${__end_idx}")
  tokens_create("${token_strings}" "${token_types}")
endmacro()
 