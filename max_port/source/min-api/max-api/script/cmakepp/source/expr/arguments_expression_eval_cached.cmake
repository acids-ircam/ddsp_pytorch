## 
##
## sets __ans in parent scope
macro(arguments_expression_eval_cached type arguments argn start end)
    arguments_expression_compile_cached("${type}" "${arguments}" "${argn}" "${start}" "${end}")
    if (__ans)
        map_tryget("${__ans}" macro_identifier)
        set(__ans "${__ans}()")
        set(__code__code "${__ans}")
        eval_ref(__code__code)
    endif ()
endmacro()
