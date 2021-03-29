
function(qm_serialize)
  # define callbacks for building result
  function(qm_obj_begin_indented)
   # message(PUSH_AFTER "qm_obj_begin_indented(${ARGN})")
    map_tryget(${context} indentation)
    ans(indentation)
    map_append_string(${context} qm "${indentation}map()\n")
    map_append_string(${context} indentation " ")
  endfunction()
  function(qm_obj_end_indented)
    #message(POP "qm_obj_end_indented(${ARGN})")
    map_tryget(${context} indentation)
    ans(indentation)
    string(SUBSTRING "${indentation}" 1 -1 indentation)
    map_set(${context} indentation "${indentation}")
    map_append_string(${context} qm "${indentation}end()\n")

  endfunction()

  function(qm_obj_keyvalue_begin_indented)
   # message("qm_obj_keyvalue_begin_indented(${key} ${ARGN}) ${context}")
    map_tryget(${context} indentation)
    ans(indentation)
    map_append_string(${context} qm "${indentation}key(\"${map_element_key}\")\n")
  endfunction()

  function(qm_literal_indented)
    map_tryget(${context} indentation)
    ans(indentation)
    
    cmake_string_escape("${node}")
    ans(node)
    map_append_string(${context} qm "${indentation} val(\"${node}\")\n")
    
    return()
  endfunction()


   map()
    kv(value              qm_literal_indented)
    kv(map_begin          qm_obj_begin_indented)
    kv(map_end            qm_obj_end_indented)
    kv(map_element_begin  qm_obj_keyvalue_begin_indented)
  end()
  ans(qm_cbs)
  function_import_table(${qm_cbs} qm_indented_callback)

  # function definition
  function(qm_serialize)        
    map_new()
    ans(context)
    map_set(${context} qm "#qm/1.0\nref()\n")
    #map_new()
    #ans(data)
    #map_set(${data} data "${ARGN}")
    dfs_callback(qm_indented_callback ${ARGN})
    map_tryget(${context} qm)
    ans(res)
    set(res "${res}end()\n")
    return_ref(res)  
  endfunction()
  #delegate
  qm_serialize(${ARGN})
  return_ans()
endfunction()



function(qm_serialize_unindented)
  # define callbacks for building result
  function(qm_obj_begin_unindented)
    map_append_string(${context} qm "map()\n")
  endfunction()
  function(qm_obj_end_unindented)
    map_append_string(${context} qm "end()\n")

  endfunction()

  function(qm_obj_keyvalue_begin_unindented)
    map_append_string(${context} qm "key(\"${map_element_key}\")\n")
  endfunction()

  function(qm_literal_unindented)
    cmake_string_escape("${node}")
    ans(node)
    map_append_string(${context} qm "val(\"${node}\")\n")
    return()
  endfunction()


   map()
    kv(value              qm_literal_unindented)
    kv(map_begin          qm_obj_begin_unindented)
    kv(map_end            qm_obj_end_unindented)
    kv(map_element_begin  qm_obj_keyvalue_begin_unindented)
  end()
  ans(qm_cbs)
  function_import_table(${qm_cbs} qm_unindented_callback)

  # function definition
  function(qm_serialize_unindented)        
    map_new()
    ans(context)
    map_set(${context} qm "#qm/1.0\nref()\n")
    #map_new()
    #ans(data)
    #map_set(${data} data "${ARGN}")
    dfs_callback(qm_unindented_callback ${ARGN})
    map_tryget(${context} qm)
    ans(res)
    set(res "${res}end()\n")
    return_ref(res)  
  endfunction()
  #delegate
  qm_serialize_unindented(${ARGN})
  return_ans()
endfunction()
