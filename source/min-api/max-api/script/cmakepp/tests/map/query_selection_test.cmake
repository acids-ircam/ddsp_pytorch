function(test)


    function(test_query_select query)
        data("${ARGN}")
        ans(data)
        obj("${query}")
        ans(query)
        timer_start(test_query_select)
        query_selection("${query}" ${data})
        ans(res)
        timer_print_elapsed(test_query_select)
        return_ref(res)
    endfunction()

    define_test_function(test_uut test_query_select query)

    test_uut("a" "{'$':{regex: {match:'(a)',replace:'$1'}}}" "a;b;c")
    test_uut("a;b" "{'[:]':{regex: 'a|b'}}" "a;b;c")
    test_uut("asd;bsd;csd" "{'$':{regex: {matchall: '[abc]sd'}}}" "asdbsdcsd")


    test_uut(
            # expected result
            "{
      a:['hello','world'],
      prop_cd:[4,5]
    }"
            # query
            "{
      a:{ regex:{matchall:'[^ ]+'}},
      'c.d[:]=>prop_cd':{where:{gt:'3'}}
    }"
            # input data
            "{
       a: 'hello world',
       b: 'goodbye world',
       c: {
        d: [1,2,3,4,5]
       } 
    }"
    )


endfunction()