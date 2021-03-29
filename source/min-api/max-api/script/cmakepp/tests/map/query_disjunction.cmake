function(test)


    function(test_query_disjunction clause)
        data("${clause}")
        ans(clause)

        data("${ARGN}")
        ans(data)

        timer_start(query_disjunction)
        query_disjunction(${clause} ${data})
        ans(res)
        timer_print_elapsed(query_disjunction)

        return_ref(res)
    endfunction()

    define_test_function(test_uut test_query_disjunction clause)


    test_uut(true "{a:['a','b']}" "{a:'b'}")
    test_uut(false "{a:['a','b']}" "{a:'c'}")
    test_uut(true "{a:['a','b'],b:'e',c:['h','f']}" "{a:'c',b:'d',c:'f'}")
    test_uut(false "{a:['a','b'],b:'e',c:['h','g']}" "{a:'c',b:'d',c:'f'}")


endfunction()