function(test)


    map_match_obj(
            "{callable: 'test_func', display_name: 'command2', id:'2', labels:'cmd2'}"
            "{callable: 'test_func', display_name: 'command3', id:'3', labels:['cmd3','cmd_3']}"
    )
    ans(res)
    assert(NOT res)

    map_match_obj("{command:'a b',args:['c','d','e']}" "{command:'a b',args:['c','d','e']}")
    ans(res)
    assert(res)

    map_match_obj("{a:[1,2,{a:3,b:5},3]}" "{a:[1,2,{b:5},3]}")
    ans(res)
    assert(res)

    map_match_obj("{a:1234}" "{a:123}")
    ans(res)
    assert(NOT res)

    map_match("123" "123")
    ans(res)
    assert(res)

    map_match("123" "234")
    ans(res)
    assert(NOT res)


    map_match_obj("" "")
    ans(res)
    assert(res)


    map_match_obj("{}" "{}")
    ans(res)
    assert(res)


    map_match_obj("{a:1,b:2,c:3}" "{a:1,b:2,c:3}")
    ans(res)
    assert(res)


    map_match_obj("{a:1,b:2,c:3}" "{a:1}")
    ans(res)
    assert(res)

    map_match_obj("{a:{aa:1,ab:2},b:{ba:{baa:1}},c:3,d:4}"
            "{a:{aa:1},b:{ba:{baa:1}},c:3}"
            )
    ans(res)
    assert(res)


endfunction()