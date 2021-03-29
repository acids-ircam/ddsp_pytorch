function(test)

    # TODO Find alternative repo to test
#    return()
#
#    timer_start(t1)
#    hg_cached_clone("https://bitbucket.com/toeb/test_repo_hg" "clone1")
#    timer_print_elapsed(t1)
#
#    timer_start(t1)
#    hg_cached_clone("https://bitbucket.com/toeb/test_repo_hg" "clone2")
#    timer_print_elapsed(t1)
#
#    return()

    timer_start(t1)
    hg_cached_clone("https://bitbucket.org/eigen/eigen" "eigen-clone1")
    timer_print_elapsed(t1)


    timer_start(t1)
    hg_cached_clone("https://bitbucket.org/eigen/eigen" "eigen-clone2")
    timer_print_elapsed(t1)

endfunction()