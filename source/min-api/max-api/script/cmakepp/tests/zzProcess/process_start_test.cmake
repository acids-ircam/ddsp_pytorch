function(test)
    ## runs three scripts and expects then to stop in a particular order
    timer_start(t1)

    process_timeout(5)
    ans(h1)

    process_timeout(10)
    ans(h2)

    process_timeout(1)
    ans(h3)

    timer_print_elapsed(t1)

    set(finished)
    function(spin)
        spinner()
        ans(current)
        echo_append("\r${ARGN}${current}")
        return()
    endfunction()

    timer_start(t1)
    message(" ")

    set(processes ${h1} ${h2} ${h3})
    while (processes)
        list_pop_front(processes)
        ans(h)
        process_isrunning(${h})
        ans(running)
        timer_elapsed(t1)
        ans(millis)
        spin("waiting ${millis} ms ")
        if (running)
            list(APPEND processes ${h})
        else ()
            list(APPEND finished ${h})
        endif ()

    endwhile ()

    echo("\rwaited ${millis} ms                        ")


    ## assert that the processes finish in order
    # TODO Doesn't work on windows 
    if(NOT WIN32)
        assert(EQUALS ${finished} ${h3} ${h1} ${h2})
    endif()
endfunction()