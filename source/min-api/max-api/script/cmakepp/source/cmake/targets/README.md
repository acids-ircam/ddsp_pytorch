## Targets

## target_list and project_list

CMake as of version 2.8.7 does not support a list of all defined targets.
Therfore I overwrote all target adding functions `add_library`, `add_executable`, `add_custom_target`, `add_test`, `install` ... which now register the name of the target globally in a list. You can access this list by using the function `target_list()` which returns the list of known target names .  Note that only targets defined before the `target_list()`  call are known.  

I did the same thing for the  `project()` command.

## target debug functions

To quickly get an overview of how your target is configured write `print_target(<target_name>)` it will print the json representation of the target as a message.

To see how were all your targetes are type `print_project_tree` which will show the json representation of all your prrojects and targets.

## target property functions

accessing target properties made easier by the following functions

* `target_get(<target> <prop-name>)` returns the value of the target property
* `target_set(<target> <prop-name> [<value> ...])` sets the value of the target property
* `target_append(<target> <prop-name> [<value> ...])` appends the values to the current value of `<prop-name>` 
* `target_has(<target> <prop-name>)->bool` returns true iff the target has a property called `<prop-name>`

