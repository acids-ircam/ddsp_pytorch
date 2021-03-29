

# {
#  pre_code
#  code
#  post_code
#  children
# }

function(ast_reduce_code ast)
    ast_reduce_code_inner("${ast}")
    ans(result)
    map_tryget("${result}" pre_code)
    ans(code)
    map_tryget("${result}" code)
    ans(next)
    set(code "${code}${next}")
    map_tryget("${result}" post_code)
    ans(next)
    set(code "${code}${next}")

  #_message("###\n${code}\n")
    return_ref(code)
endfunction()

function(ast_reduce_code_inner ast)
    map_tryget("${ast}" pure_value)
    ans(is_pure_value)

    ## pure values do not produce any code
    if (is_pure_value)
        return()
    endif ()

    set(pre_code)
    set(code)
    set(post_code)

    ## get code for all children
    map_tryget("${ast}" children)
    ans(children)
    foreach (child ${children})
        ast_reduce_code_inner("${child}")
        ans(current_result)

        map_tryget("${current_result}" pre_code)
        ans(current_pre_code)
        set(pre_code "${pre_code}${current_pre_code}")
        map_tryget("${current_result}" code)
        ans(current_code)
        set(code "${code}${current_code}")
        map_tryget("${current_result}" post_code)
        ans(current_post_code)
        set(post_code "${post_code}${current_post_code}")
    endforeach ()

    #print_vars(--plain  pre_code code post_code )

    map_tryget("${ast}" code)
    ans(current_code)

    set(code "${pre_code}${code}${current_code}${post_code}")


    map_tryget("${ast}" pre_code)
    ans(pre_code)
    map_tryget("${ast}" post_code)
    ans(post_code)

    map_new()
    ans(result)
    map_set(${result} pre_code "${pre_code}")
    map_set(${result} code "${code}")
    map_set(${result} post_code "${post_code}")

    return_ref(result)
endfunction()

