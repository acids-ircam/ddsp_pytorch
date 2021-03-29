#adds a values to parent_scopes __ans
function(yield)
    set(__yield_tmp ${__yield_tmp} ${ARGN} PARENT_SCOPE)

endfunction()