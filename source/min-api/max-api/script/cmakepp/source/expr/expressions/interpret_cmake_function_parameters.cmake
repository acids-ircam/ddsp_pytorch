
function(interpret_cmake_function_parameters tokens)
  

  interpret_elements("${tokens}" "comma" "interpret_key_value;interpret_rvalue")
  ans(members)


  next_id()
  ans(ref)
  set(value "\${${ref}}")
  set(code "map_new()\nans(${ref})\n")
  foreach(member ${members})
    map_tryget("${member}" expression_type)
    ans(member_type)
    if("${member_type}" STREQUAL "key_value")
      map_tryget("${member}" key_ast)
      ans(member_key_ast)
      map_tryget("${member_key_ast}" value )
      ans(member_key_ast_value)
      map_tryget("${member}" value_ast)
      ans(member_value_ast)
      map_tryget("${member_value_ast}" value)
      ans(member_value_ast_value)
      set(code "${code}map_set(${value} \"${member_key_ast_value}\" \"${member_value_ast_value}\")\n")
    elseif("${member_type}" STREQUAL "ellipsis")
      map_tryget("${member}" value)
      ans(member_value)
      set(code "${code}address_append(${value} \"${member_value}\")\n")      
    else() ## normal rvalue      
      map_tryget("${member}" value)
      ans(member_value)
      set(code "${code}encoded_list(\"${member_value}\")\naddress_append(${value} \"\${__ans}\")\n")      
    endif()

  endforeach()


  ast_new(
    "${tokens}"          # tokens
    "object"            # expression_type
    "object"            # value_type
    "${ref}"            # ref
    "${code}"           # code
    "${value}"          # value
    "false"             # const
    "false"             # pure_value
    "${members}"        # children
    )
  ans(ast)

  return_ref(ast)

endfunction()

