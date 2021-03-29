function(test)


    function(test_query_match query)

        data("${ARGN}")
        ans(data)
        data("${query}")
        ans(query)

        timer_start(query_match_cnf)
        query_match_cnf("${query}" "${data}")
        ans(res)
        timer_print_elapsed(query_match_cnf)


        return_ref(res)
    endfunction()

    define_test_function(test_uut test_query_match query)

    test_uut("true" "{a:['a','b'], b:'d' };{c:'d'};{d:['e','f']}" "{a:'b',c:'d',d:'f'}")
    test_uut("false" "{a:['a','b'], b:'d' };{c:'d'};{d:['k','p']}" "{a:'b',c:'d',d:'f'}")


endfunction()