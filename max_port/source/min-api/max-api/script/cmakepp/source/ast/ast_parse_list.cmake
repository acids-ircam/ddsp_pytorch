

 function(ast_parse_list )#definition stream create_node
 
   # message("parsing list")
    token_stream_push(${stream})

    map_tryget(${definition}  begin)
    ans(begin)
    map_tryget(${definition}  end)
    ans(end)
    map_tryget(${definition}  separator)
    ans(separator)
    map_get(${definition}  element)
    ans(element)
   # message(" ${begin} <${element}> <${separator}> ${end}")
    
    #message("create node ${create_node}")
    if(begin)
      ast_parse(${stream} ${begin})
      ans(begin_ast)
      
      if(NOT begin_ast)
        token_stream_pop(${stream})
        return(false)
      endif()

    endif()
    set(child_list)
    while(true)
      if(end)
        ast_parse(${stream} ${end})
        ans(end_ast)
        if(end_ast)
          break()
        endif()
      endif()

      if(separator)
        if(child_list)
          ast_parse(${stream} ${separator})
          ans(separator_ast)

          if(NOT separator_ast)
            token_stream_pop(${stream})
          #  message("failed")
            return(false)
          endif()
        endif()
      endif()
      
      ast_parse(${stream} ${element})
      ans(element_ast)

      if(NOT element_ast)
        #failed because no element was found
        if(NOT end)
          break()
        endif()
        return(false)
      endif()
      list(APPEND child_list ${element_ast})

     # message("appending child ${element_ast}")

      

    endwhile()
    #message("done ${create_node}")
    token_stream_commit(${stream})

    if(NOT create_node)
      return(true)
    endif()
#    message("creating node")

    is_map("${begin_ast}" )
    ans(isnode)
    if(NOT isnode)
      set(begin_ast)
    endif()
    is_map("${end_ast}" )
    ans(isnode)
    if(NOT isnode)
      set(end_ast)
    endif()
    map_tryget(${definition}  name)
    ans(def)
    map_new()
    ans(node)
    map_set(${node} types ${def})
    map_set(${node} children ${begin_ast} ${child_list} ${end_ast})
    return(${node})
  endfunction()