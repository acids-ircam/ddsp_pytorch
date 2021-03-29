function(test)

    timer_start(timer1)
    git_cached_clone("https://github.com/AnotherFoxGuy/cmakepp" "dir1")
    timer_print_elapsed(timer1)

    timer_start(timer2)
    git_cached_clone("https://github.com/AnotherFoxGuy/cmakepp" "dir2")
    timer_print_elapsed(timer2)

    timer_start(timer3)
    git_cached_clone("https://github.com/AnotherFoxGuy/cmakepp" "dir2" --read package.cmake)
    ans(res)
    assert(res)
    timer_print_elapsed(timer3)

    message("${res}")

endfunction()