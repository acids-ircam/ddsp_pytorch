

function(eval_predicate_template_cmake scope template)
    template_run_scoped("${scope}" "${template}")
    ans(expr)
    eval_predicate_cmake("${expr}")
    ans(use)        
    return_ref(use)
endfunction()