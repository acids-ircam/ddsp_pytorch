

 function(map_template_evaluate_scoped scope)
 #todo -> needs to be implemented to correctly do recursion
 ## needs to have max iterations
    if(NOT __recurse)
        set(__recurse true)
        map_new()
        ans(visited)
    endif()

    list(LENGTH scope count)

    if("${count}" EQUAL "0")
        return()
    endif()


    if("${count}" GREATER "1")
        set(result)
        foreach(arg ${ARGN})
            map_template_evaluate_scoped("${arg}")
            ans_append(result)            
        endforeach()
        return_ref(result)
    endif()

    set(value ${ARGN})
    is_map("${value}")
    ans(is_map)
    if(NOT is_map)
        template_run_scoped("${scope}" "${value}")
        return_ans()
    endif()

    map_has("${visited}" "${value}")
    ans(result)

    if(result)
        return_ref(result)
    endif()
    

    map_new()
    ans(result)

    map_set(${visited} "${value}" "${result}")

    map_keys("${value}")
    ans(keys)
        
    foreach(key ${keys})
        map_tryget("${value}" "${key}")
        ans(propTemplate)
        map_template_evaluate_scoped("${scope}" ${propTemplate})
        ans(prop)
        map_set("${result}" "${key}" "${prop}")
    endforeach()

    return_ref(result)

        # map_clone_deep(${})
        # ans(clone)


        # if("${value}" STREQUAL "${scope}")
        #     set(scope "${clone}")
        # endif()
            
        # set(changed true)
        # while(changed)




        #     set(changed false)
        #     foreach(key ${keys})
        #         map_tryget(${value} "${key}")
        #         ans(template)
        #         map_tryget(${clone} "${key}")
        #         ans(previousValue)

        #         template_run_scoped(${scope} "${template}" )
        #         ans(value)
        #         print_vars(key template value)

        #         if(NOT "${value}_" STREQUAL "${previousValue}_")
        #             map_set(${clone} "${key}" "${value}")
        #             set(changed true)
        #         endif()
        #     endforeach()
        # endwhile()



# if(changed)
#         return()
#     else()
#         return(${clone})
#     endif()

    return_ref(result)
    
 endfunction()