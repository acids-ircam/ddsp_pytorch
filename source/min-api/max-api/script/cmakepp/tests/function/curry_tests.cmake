function(test)
    curry3(() => invocation_argument_string)
    ans(res)
    call("${res}" (hello you dude))
    ans(res)
    assert("${res}" STREQUAL "hello you dude")

    curry3(() => invocation_argument_string (asd /1))
    ans(res)
    call("${res}" (bsd csd dsd))
    ans(res)
    assert("${res}" STREQUAL "asd csd")

    curry3(abc => invocation_argument_string)
    assert(COMMAND abc)
    curry3(abc2 () => invocation_argument_string)
    assert(COMMAND abc2)
    curry3(abc3 (a) => invocation_argument_string)
    assert(COMMAND abc3)
    curry3(abc4 (a b) => invocation_argument_string)
    assert(COMMAND abc3)
    curry3(invocation_argument_string)
    ans(res)
    assert(res)
    call("${res}" (hello))
    ans(res)
    assert("${res}" STREQUAL "hello")
    curry3(=> invocation_argument_string)
    ans(res)
    call("${res}" (hello you))
    ans(res)
    assert("${res}" STREQUAL "hello you")


    curry3(() => address_set_new ("/0"))
    ans(res)
    assert(res)

    curry3(invocation_argument_string)
    ans(res)
    call("${res}" (1 2 3))
    ans(res)
    assert("${res}" STREQUAL "1 2 3")

    curry3(invocation_argument_string (2 3 /1))
    ans(res)
    call("${res}" (a b c))
    ans(res)
    assert("${res}" STREQUAL "2 3 b")


    curry3((a b c) => invocation_argument_string (/c /b /a))
    ans(res)
    call("${res}" (1 2 3))
    ans(res)
    assert("3 2 1" STREQUAL "${res}")

    curry3(myfunc (a b c) => invocation_argument_string ("ab;c" /* /c))
    ans(res)

    myfunc(1 2 3 4 5)
    ans(res)
    assert("${res}" EQUALS "\"ab;c\" 4 5 3")

    function(test_curry)
        return(${a} ${b} ${c} ${ARGN})
    endfunction()
    set(a 123)
    set(b 234)
    set(c 345)
    curry3(myfunc (a b c) => [a;b] test_curry (/3 /4 /4 /a /c /b "1;2;3" /*))

    set(a 456)
    set(c 567)

    myfunc(9 8 7 p o i)
    ans(res)

    assert(${res} EQUALS 123 234 567 p o o 9 7 8 1 2 3 p o i)

    # takes about 30 ms per curry call
    timer_start(curry_compile_100)
    foreach (i RANGE 0 100)
        curry3(myfunc (a b c) => invocation_argument_string ("ab;c" /* /c))
    endforeach ()
    timer_print_elapsed(curry_compile_100)


    curry3([a;b;c] string_length (/c /b /a) => myfunc (a b c))
    ans(res)

    curry3([a;b;c] "[](a)message({{a}})" (/c /b /a) => myfunc (a b c))
    ans(res)

    curry3(string_length ("a;b;c;d") => myfunc (a b c))
    ans(res)


    function(funcA a b c)
        return("${a}${b}${c}")
    endfunction()


    curry3(funcB => funcA (/0 44 /1))
    funcB("1" "2")
    ans(res)
    assert("${res}" STREQUAL 1442)


    curry3(funcB () => funcA (/1 nana /0))
    funcB("1" "2")
    ans(res)
    assert("${res}" STREQUAL 2nana1)

    function(funcC var1 var2)
        return("${var1}${var2}${var3}${var4}")
    endfunction()
    set(var3 3)
    set(var4 4)

    bind(funcC var3 var4 as funcB)

    set(var3 5)
    set(var4 6)


    funcB(1 2)
    ans(res)
    assert("${res}" STREQUAL "1234")

endfunction()