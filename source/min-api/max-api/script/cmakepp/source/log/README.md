## Logging Functions


`CMake`'s logging is restricted to using the built in `message()` function. It writes the messages to `stdout` and `stderr` depending on the given tag present (`STATUS`, `ERROR`, `FATAL_ERROR`,`WARNING`, `<none>`).  This is sometimes not enough - especially when the output of your `CMake` script should be very controlled (ie. it is important that no debug or status messages are ouput when users expect the output to adher to a certain format)

This is why I started to write log functions which do not output anything.  You can listen to log messages using the `event` system - the `on_log_message` is called for every log message that is output.


### Function List


* [error](#error)
* [fatal](#fatal)
* [log](#log)
* [log_record_clear](#log_record_clear)
* [log_default_handler](#log_default_handler)
* [log_last_error_entry](#log_last_error_entry)
* [log_last_error_message](#log_last_error_message)
* [log_last_error_print](#log_last_error_print)
* [log_print](#log_print)
* [warning](#warning)

### Function Descriptions

## <a name="error"></a> `error`

 `error(...)-><log entry>`

 Shorthand function for `log(<message> <refs...> --error)
 
 see [log](#log)





## <a name="fatal"></a> `fatal`

 reports an error and stops program exection 




## <a name="log"></a> `log`

 `log(<message:<string>> <refs...> [--error]|[--warning]|[--info]|[--debug]) -> <void>`

 This is the base function on which all of the logging depends. It transforms
 every log message into a object which can be consumed by listeners or filtered later

 *Note*: in its current state this function is not ready for use

 * returns
   * the reference to the `<log entry>`
 * parameters
   * `<message>` a `<string>` containing the message which is to be logged the data may be formatted (see `format()`)
   * `<refs...>` you may pass variable references which will be captured so you can later check the state of the application when the message was logged
 * flags
   * `--error`    flag indicates that errors occured
   * `--warning`  flag indicates warnings
   * `--info`     flag indicates a info output
   * `--debug`    flag indicates a debug output
 * values
   * `--error-code <code>` 
   * `--level <n>` 
   * `--push <section>` depth+1
   * `--pop <section>`  depth-1
 * events
   * `on_log_message`

 *Examples*
 ```
 log("this is a simple error" --error) => null
 ```




## <a name="log_record_clear"></a> `log_record_clear`

 `log_record_clear()-><void>`
 
 removes all messages from the log record






## <a name="log_default_handler"></a> `log_default_handler`





## <a name="log_last_error_entry"></a> `log_last_error_entry`

 `log_last_error_entry()-><log entry>`

 returns the last log entry which is an error
 




## <a name="log_last_error_message"></a> `log_last_error_message`

 `log_last_error_message()-><string>`

 returns the last logged error message





## <a name="log_last_error_print"></a> `log_last_error_print`

 `log_last_error_print()-><void>`

 prints the last error message to the console  





## <a name="log_print"></a> `log_print`

 `log_print`






## <a name="warning"></a> `warning`







