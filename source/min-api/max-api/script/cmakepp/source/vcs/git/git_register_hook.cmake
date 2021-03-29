# registers a git hook
function(git_register_hook hook_name)
  git_directory()
  ans(git_dir)


endfunction()


function(git_local_hooks)
  set(hooks
    pre-commit
    post-commit
    prepare-commit-msg
    commit-msg
    pre-rebase
    post-checkout

    )
  return_ref(hooks)

endfunction()
