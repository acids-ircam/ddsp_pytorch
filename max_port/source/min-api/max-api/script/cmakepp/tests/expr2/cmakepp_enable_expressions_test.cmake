function(test)

    # TODO This test fails because the build/script.cmake is being evaluated instead of this file...
#    function(testf)
#        set(result "$['hello']")
#        cmakepp_enable_expressions("${CMAKE_CURRENT_LIST_LINE}")
#        set(result "${result}$['hello']")
#        return_ref(result)
#    endfunction()
#
#    testf()
#    ans(res)
#    assert("${res}" STREQUAL "$['hello']hello")


    # function(testf2)
    #   ## enables expressions in current scope (scope being a file or a function)
    #   cmakepp_enable_expressions("${CMAKE_CURRENT_LIST_LINE}")
    #   message("so this is a test to look if it works $[test::string_length()]")
    #   return("$[test::string_length()]")
    # endfunction()

    # message("so this is a test to look if it works $[test::string_length()]")

    # testf2()
    # ans(res)

    # assert("${res}" EQUAL 4)


endfunction()