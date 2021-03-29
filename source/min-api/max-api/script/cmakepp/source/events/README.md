## Events

Events are often usefull when working with modules. CMake of course has no need for events generally. Some of my projects (cutil/cps) needed them however. For example the package manager cps uses them to allow hooks on package install/uninstall/load events which the packages can register.


## Example


```
# create an event handler
function(my_event_handler arg)
 message("${event_name} called: ${arg}")
 return("answer1")
endfunction()

# add an event handler
event_addhandler(irrelevant_event_name my_event_handler)
# add lambda event handler
event_addhandler(irrelevant_event_name "(arg)->return($arg)")
# anything callable can be used as an event handler (even a cmake file containing a single function)

# emit event calls all registered event handlers in order
# and concatenates their return values
# side effects: prints irrelevent_event_name called: i am an argument
event_emit(irrelevant_event_name "i am an argument")
ans(result)


assert(EQUALS ${result} answer1 "i am an argument")
```

## Functions and Datatypes

* `<event id>` a globally unique identifier for an event (any name you want except `on_event` )
* `on event` a special event that gets fired on all events (mainly for debugging purposes)
* `event_addhandler(<event id> <callable>)` registers an event handler for the globally unique event identified by `<event id>` see definition for callable in [functions section](#functions)
* `event_removehandler(<event id> <callable>)` removes the specified event handler from the handler list (it is no longer invoked when event is emitted)
* `event_emit(<event id> [arg ...]) -> any[]` invokes the event identified by `<event id>` calls every handler passing along the argument list. every eventhandler's return value is concatenated and returned.  It is possible to register event handlers during call of the event itself the emit routine continues as long as their are uncalled registered event handlers but does not call them twice.
* ... (functions for dynamic events, access to all available events)




### Function List


* [event](#event)
* [event_addhandler](#event_addhandler)
* [event_cancel](#event_cancel)
* [event_clear](#event_clear)
* [event_emit](#event_emit)
* [event_get](#event_get)
* [event_handler](#event_handler)
* [event_handler_call](#event_handler_call)
* [event_handlers](#event_handlers)
* [event_new](#event_new)
* [event_removehandler](#event_removehandler)
* [events](#events)
* [events_track](#events_track)
* [is_event](#is_event)


### Function Descriptions

## <a name="event"></a> `event`

 `(<event-id>):<event>`

 tries to get the `<event>` identified by `<event-id>`
 if it does not exist a new `<event>` is created by  




## <a name="event_addhandler"></a> `event_addhandler`

 `event_addhandler(<~event> <~callable>)-><event handler>`

 adds an event handler to the specified event. returns an `<event handler>`
 which can be used to remove the handler from the event.





## <a name="event_cancel"></a> `event_cancel`

 `()-><null>`

 only usable inside event handlers. cancels the current event and returns
 after this handler.




## <a name="event_clear"></a> `event_clear`

 `(<~event>)-><void>`

 removes all handlers from the specified event




## <a name="event_emit"></a> `event_emit`

 `(<~event> <args:<any...>>)-><any...>`

 emits the specified event. goes throug all event handlers registered to
 this event and 
 if event handlers are added during an event they will be called as well

 if a event calls event_cancel() 
 all further event handlers are disregarded

 returns the accumulated result of the single event handlers




## <a name="event_get"></a> `event_get`

 `(<~event>)-><event>`
  
 returns the `<event>` identified by `<event-id>` 
 if the event does not exist `<null>` is returned.




## <a name="event_handler"></a> `event_handler`

 `(<~callable>)-><event handler>` 

 creates an <event handler> from the specified callable
 and returns it. a `event_handler` is also a callable




## <a name="event_handler_call"></a> `event_handler_call`

 `(<event> <event handler>)-><any>`

 calls the specified event handler for the specified event.




## <a name="event_handlers"></a> `event_handlers`

 `(<event>)-><event handler...>`

 returns all handlers registered for the event




## <a name="event_new"></a> `event_new`

 `(<?event-id>)-><event>`

 creates an registers a new event which is identified by
 `<event-id>` if the id is not specified a unique id is generated
 and used.
 
 returns a new <event> object: 
 {
   event_id:<event-id>
   handlers: <callable...> 
   ... (psibbly cancellable, aggregations)
 }
 also defines a global function called `<event-id>` which can be used to emit the event





## <a name="event_removehandler"></a> `event_removehandler`

 `(<event handler>)-><bool>`

 removes the specified handler from the event idenfied by event_id
 returns true if the handler was removed




## <a name="events"></a> `events`

 `()-> <event>`

 returns the global events map it contains all registered events.




## <a name="events_track"></a> `events_track`

 `(<event-id...>)-><event tracker>`

 sets up a function which listens only to the specified events
 




## <a name="is_event"></a> `is_event`

 `(<any>)-><bool>`

 returns true if the specified value is an event
 an event is a ref which is callable and has an event_id






