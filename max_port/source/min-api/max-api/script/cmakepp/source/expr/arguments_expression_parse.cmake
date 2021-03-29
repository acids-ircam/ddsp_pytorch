
macro(arguments_expression_parse type arguments start end)
    arguments_create_tokens("${start}" "${end}")
    ans(tokens)
    map_new()
    ans(context)
    map_set(${context} current_id 0)
    ## stupid variable expansion
    set(____code ${arguments})
    set(____code "${type}(\"${tokens}\" ${____code})")
    eval("${____code}")
endmacro()
