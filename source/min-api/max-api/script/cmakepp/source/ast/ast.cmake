# parses an abstract syntax tree from str
function(ast str language)
  language("${language}")
  ans(language)
  # set default root definition to expr
  set(root_definition ${ARGN})
  if(NOT root_definition)
    
    map_get("${language}"  root_definition)
    ans(root_definition)
  endif()



  # transform str to a stream
  token_stream_new(${language} "${str}")
  ans(stream)
  # parse ast and return result
  ast_parse(${stream} "${root_definition}" ${language})
  return_ans()
endfunction()