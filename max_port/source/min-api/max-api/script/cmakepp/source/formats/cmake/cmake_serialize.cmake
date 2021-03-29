
  function(cmake_serialize)
      function(cmake_ref_format)
        set(prop)
        if(ARGN)
          set(prop ".${ARGN}")
        endif()
        set(__ans ":\${ref}${prop}" PARENT_SCOPE)
      endfunction()

     # define callbacks for building result
    function(cmake_obj_begin)
      map_tryget(${context} ${node})
      ans(ref)
      map_push_back(${context} refstack ${ref})
      map_append_string(${context} qm 
"math(EXPR ref \"\${base} + ${ref}\")
set_property(GLOBAL PROPERTY \":\${ref}.__keys__\" \"\")
")
    endfunction()

    function(cmake_obj_end)
      map_pop_back(${context} refstack)
      map_peek_back(${context} refstack)
      ans(ref)

      map_append_string(${context} qm 
"math(EXPR ref \"\${base} + ${ref}\")
")
    endfunction()
    
    function(cmake_obj_keyvalue_begin)
      cmake_ref_format()
      ans(keystring)
      cmake_ref_format(${map_element_key})
      ans(refstring)
      
      map_append_string(${context} qm 
"set_property(GLOBAL APPEND PROPERTY \"${keystring}.__keys__\" \"${map_element_key}\")
set_property(GLOBAL PROPERTY \"${refstring}\")
")
    endfunction()

    function(cmake_literal)
      cmake_ref_format(${map_element_key})
      ans(refstring)
      cmake_string_escape("${node}")
      ans(node)
      map_append_string(${context} qm 
"set_property(GLOBAL APPEND PROPERTY \"${refstring}\" \"${node}\")
")
      return()
    endfunction()

    function(cmake_unvisited_reference)
      map_tryget(${context} ref_count)
      ans(ref_count)
      math(EXPR ref "${ref_count} + 1")
      map_set_hidden(${context} ref_count ${ref})
      map_set_hidden(${context} ${node} ${ref})

      cmake_ref_format(${map_element_key})
      ans(refstring)

      map_append_string(${context} qm
"math(EXPR value \"\${base} + ${ref}\")
set_property(GLOBAL PROPERTY \":\${value}.__type__\" \"map\")
set_property(GLOBAL APPEND PROPERTY \"${refstring}\" \":\${value}\")
")
    endfunction()
    function(cmake_visited_reference)
map_tryget(${context} "${node}")
ans(ref)

  cmake_ref_format(${map_element_key})
  ans(refstring)
map_append_string(${context} qm
"#revisited node
math(EXPR value \"\${base} + ${ref}\")
set_property(GLOBAL APPEND PROPERTY \"${refstring}\" \":\${value}\")
# end of revisited node
")


    endfunction()
     map()
      kv(value              cmake_literal)
      kv(map_begin          cmake_obj_begin)
      kv(map_end            cmake_obj_end)
      kv(map_element_begin  cmake_obj_keyvalue_begin)
      kv(visited_reference  cmake_visited_reference)
      kv(unvisited_reference  cmake_unvisited_reference)
    end()
    ans(cmake_cbs)
    function_import_table(${cmake_cbs} cmake_callback)

    # function definition
    function(cmake_serialize)        
      map_new()
      ans(context)
      map_set(${context} refstack 0)
      map_set(${context} ref_count 0)
  
      dfs_callback(cmake_callback ${ARGN})
      map_tryget(${context} qm)
      ans(res)
      map_tryget(${context} ref_count)
      ans(ref_count)

      set(res "#cmake/1.0
get_property(base GLOBAL PROPERTY \":0\")
set(ref \${base})
${res}math(EXPR base \"\${base} + ${ref_count} + 1\")
set_property(GLOBAL PROPERTY \":0\" \${base})
")

      return_ref(res)  
    endfunction()
    #delegate
    cmake_serialize(${ARGN})
    return_ans()
  endfunction()
