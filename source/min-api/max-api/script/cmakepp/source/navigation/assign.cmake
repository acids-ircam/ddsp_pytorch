## `([!]<expr> <value>|("="|"+=" <expr><call>)) -> <any>`
##
## the assign function allows the user to perform some nonetrivial 
## operations that other programming languages allow 
##
## Examples
## 
function(assign __lvalue __operation __rvalue)
    ## is a __value

    if (NOT "${__operation}" MATCHES "^(=|\\+=)$")
        ## if no equals sign is present then interpret all
        ## args as a simple literal cmake value
        ## this allows the user to set an expression to
        ## a complicated string with spaces without needing
        ## to single quote it
        set(__value ${__operation} ${__rvalue} ${ARGN})
    elseif (${__rvalue} MATCHES "^'.*'$")
        string_decode_delimited("${__rvalue}" ')
        ans(__value)
    elseif (${__rvalue} MATCHES "(^{.*}$)|(^\\[.*\\]$)")
        script("${__rvalue}")
        ans(__value)
    else ()
        navigation_expression_parse("${__rvalue}")
        ans(__rvalue)
        list_pop_front(__rvalue)
        ans(__ref)

        if ("${ARGN}" MATCHES "^\\(.*\\)$")
            ref_nav_get("${${__ref}}" "%${__rvalue}")
            ans(__value)

            map_tryget(${__value} ref)
            ans(__value_ref)

            data("${ARGN}")
            ans(__args)
            if (NOT __value_ref)
                call("${__ref}" ${__args})
                ans(__value)

            else ()
                map_tryget(${__value} property)
                ans(__prop)
                map_tryget(${__value} range)
                ans(ranges)

                if (NOT ranges)
                    list_pop_front(__args)
                    list_pop_back(__args)
                    obj_member_call("${__value_ref}" "${__prop}" ${__args})
                    ans(__value)

                else ()
                    map_tryget(${__value} __value)
                    ans(__callables)
                    set(__value)
                    set(this "${__value_ref}")
                    foreach (__callable ${__callables})
                        call("${__callable}" ${__args})
                        ans(__res)
                        list(APPEND __value ${__res})
                    endforeach ()
                endif ()
            endif ()
        else ()
            ref_nav_get("${${__ref}}" ${__rvalue})
            ans(__value)
        endif ()
    endif ()
    string_take(__lvalue !)
    ans(__exc)
    navigation_expression_parse("${__lvalue}")
    ans(__lvalue)
    list_pop_front(__lvalue)
    ans(__lvalue_ref)

    if ("${__operation}" STREQUAL "+=")
        ref_nav_get("${${__lvalue_ref}}" "${__lvalue}")
        ans(prev_value)
        set(__value "${prev_value}${__value}")
    endif ()
    # message("ref_nav_set ${${__lvalue_ref}} ${__exc}${__lvalue} ${__value}" )
    ref_nav_set("${${__lvalue_ref}}" "${__exc}${__lvalue}" "${__value}")
    ans(__value)
    set(${__lvalue_ref} ${__value} PARENT_SCOPE)
    return_ref(__value)
endfunction()