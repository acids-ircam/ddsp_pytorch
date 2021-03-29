function(test)

    set(name "Edgar")
    template_run("Hello this is <%={name}%>")
    ans(generated_content)
    assert("${generated_content}" STREQUAL "Hello this is Edgar")

    function(my_echo_func inp)
        return("${inp}")
    endfunction()

    function(my_echo_func_2_args inp inpTwo)
        return("#${inpTwo} ${inp}")
    endfunction()

    template_run("My echo test returns: <%= my_echo_func('Hello World') %>")
    ans(generated_content)
    assert("${generated_content}" STREQUAL "My echo test returns: Hello World")

    template_run("My echo test returns: <%= my_echo_func_2_args('World', 'Hello') %>")
    ans(generated_content)
    assert("${generated_content}" STREQUAL "My echo test returns: #Hello World")


    fwrite_temp("My echo test returns from file: <%= my_echo_func('Hello World') %>" ".in")
    ans(temppath)
    template_run_file("${temppath}")
    ans(generated_content)
    assert("${generated_content}" STREQUAL "My echo test returns from file: Hello World")

    fwrite_temp("My echo test returns from file: @my_echo_func('Hello World using short notation')" ".in")
    ans(temppath)
    template_run_file("${temppath}")
    ans(generated_content)
    assert("${generated_content}" STREQUAL "My echo test returns from file: Hello World using short notation")

endfunction()