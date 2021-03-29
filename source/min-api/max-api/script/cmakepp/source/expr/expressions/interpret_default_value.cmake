 function(interpret_default_value tokens)
    set(lhs_tokens ${tokens})

    list_pop_back(lhs_tokens)
    ans(question_mark_token)

    if(NOT question_mark_token)
      throw("no tokens" --function interpret_default_value)
    endif()


    map_tryget("${question_mark_token}" type)
    ans(question_token_type)
    if(NOT "${question_token_type}" STREQUAL "question_mark")
      ans(question_mark_token)
      throw("expected an question token, got ${question_token_type}")
    endif()


    interpret_lvalue("${lhs_tokens}")
    rethrow()
    ans(lvalue)
      
    map_tryget("${lvalue}" ref)
    ans(lvalue_ref)
    map_tryget("${lvalue}" value)
    ans(lvalue_value)

    set(code "is_address(\"${lvalue_value}\")\nif(NOT __ans)\nmap_new()\nans(${lvalue_ref})\nendif()\n")

    ast_new(
      "${tokens}"
      "reference_coercion"
      "address"
      "${lvalue_ref}"
      "${code}"
      "\${${lvalue_ref}}"
      "false"
      "false"
      "${lvalue}"
      )
    ans(ast)
    map_set("${ast}" post_code "${post_code}")
    return_ref(ast)
  endfunction()
