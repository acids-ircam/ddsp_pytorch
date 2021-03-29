
    function(git_scm_descriptor git_ref)

        set(scm_descriptor)
        assign(!scm_descriptor.scm = 'git')
        assign(!scm_descriptor.ref = git_ref)

        return_ref(scm_descriptor)
    endfunction()