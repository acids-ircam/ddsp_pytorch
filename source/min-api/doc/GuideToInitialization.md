# Initialization

Min objects are external objects for Max written in C++ using a high-level declarative application programming interface.  To get started refer to [Writing Min Objects](GuideToWritingObjects.md).



## Order of Initialization

The order in which calls and assignments are made when you create a new instance of your object is detailed here. In some cases, careful thought will need to be given to how your members are initialized if they have cross-dependencies upon each other.

### First: Members are Initialized from Top to Bottom

Any members of your class are initialized in the order in which they appear in the class definition. These members may be "plain old data" variables representing an int or a float, or they may be other classes such as inlets, outlets, queues, timers, buffer_references, messages, or attributes.

Among these **Attributes** are a special case. Whereas other types may provide callback functions that are executed in response to actions in the Max environment, the functions provided for setting attributes are called as a part of setting the attribute to its initial value.

Here is an annotated example:

```c++
class my_object : public object<my_object> {
public:
	inlet<>	in { this, "(toggle) on/off" }; // initialized first, but does nothing
	outlet<> out { this, "(bang) output" }; // initialized second, but does nothing

	// initialized third, does nothing on its own -- BUT SEE BELOW
	timer metro { this,
    	MIN_FUNCTION {
			out.send("bang");
			metro.delay(interval);
			return {};
		}
    };

  	// initialized fourth, the setter is called as the value is set to 250.0
    // the setter is not calling any other members of this class, 
    // so it doesn't really matter where in the class order this appears.
	attribute<number> minimum { this, "minimum", 250.0,
		title { "Minimum Interval" },
		description { "Lower-bound of generated random interval." },
		setter { 
          	MIN_FUNCTION {
				double value = args[0];
			
				if (value < 1.0)
					value = 1.0;
				return {value};
			}
        },
	};

  	// initialized fifth, sets the timer to be off
    // IMPORTANT: because the setter calls a method of the 'metro'
    // member, that timer must be defined (so that it is initialized)
    // prior to the definition of this attribute!
	attribute<bool> on { this, "on", false,
		title { "On/Off" },
		description { "Activate the timer." },
		setter { 
       		MIN_FUNCTION {
				if (args[0] == true)
					metro.delay(0.0);	// fire the first one straight-away
				else
					metro.stop();
				return args;
			}
        }
	};

	// initialized sixth, just a message and it isn't called from anywhere else
    // so this could be located anywhere in the initialization order.
	message<> toggle { this, "int", "Toggle the state of the timer.",
		MIN_FUNCTION {
			on = args[0];
			return {};
		}
	};
};

MIN_EXTERNAL(my_object);
```



### Second: Constructor is Called

*If there is a constructor*, it is passed all of the non-attribute arguments that a user has typed into your object's box. It is called after all of the members are initialized.

### Third: Arguments are Processed

*If there is no constructor*, and there are argument declarations, those arguments are handled in order from first to last. Each will call their function as non-attribute arguments are processed that a user has typed into your object's box.

### Fourth: Attributes Arguments are Processed

Attribute values entered into the Max object box are processed last — after the member initialization, constructor, and other argument handling. For each attribute value that a user has typed into your object's box, that attribute will be set to this new value.



## Complex Class Initialization

Below is an example of a more complex scenario for class initialization. In this scenario we have a member which is a pointer to an instance that performs some special task (in this case it generates beats). The attributes of the object make calls to the instance owned by the pointer, so the pointer must be valid prior to the attributes calling it during initialization. But the pointer cannot be allocated until the custom constructor is called because we need to pass an argument to it that the user types into the object box. And remember that the constructor is not called until *after* all members are initialized!

```c++
class beat : public object<beat> {
private:
  
    // initialized first!
  	// CRITICAL because other member initialization below relies on this value!
	bool m_initialized { false };

public:

  	// initializing inlets and outlets second, not consequential.
	inlet<>		input		{ this, "(toggle) turn the metro on/off" };
    outlet<>	bar_sync	{ this, "(bang) beats" };
  
	// constructor will be called AFTER *all* of the members
    // (most of which are located below in this class definition).
	beat(const atoms& args = {}) {
      
        // processing our arguments makes an assignment to the
        // tempo attribute -- 
        // okay because it will have already been initialized with the members
		if (args.size() > 0)
			tempo = args[0];
		
      	// create an instance of "beat_generator" which is a class defined
        // in a library or in another piece of code we authored elsewhere.
		m_beat_generator = std::make_unique<beat_generator>(tempo);
      
      	// now that m_beat_generator is valid and our object is properly
        // initialized, we can switch our flag used to prevent unsafe access
        // in the attribute setters (below)
		m_initialized = true;
	}
	

	attribute<double> tempo { this, "tempo", 120.0,
		setter { MIN_FUNCTION {
          	// only run this part of the setter if we are initialized!
            // doing otherwise results in unsafe access of an uninitialized member
            // pointer and will most likely crash
			if (m_initialized) {
				double newtempo = args[0];				
				m_beat_generator.setTempo(newtempo);
			}
          
          	// we still return the args, which will result in the internal 
          	// data member being assigned 120.0 when the attribute is initialized.
			return args;
		}}
	};
	
	
  	// This message defines a function that calls the timer member called "runner".
  	// That member has not yet been initialized, but the message's function
  	// will not be called as a part of initialization -- so this is not a problem.
	message<> toggle { this, "int", "Turn on/off the output of beats. ",
		MIN_FUNCTION {
			bool on = args[0];
			if (on)
				runner.delay(0.0);
			else
				runner.stop();
			return {};
		}
	};

	// The timer is initialized almost last
    // No calls are made to it until a user sends messages to the object
    // so it could be initialized anywhere in the order.
	timer runner { this, 
		MIN_FUNCTION {
      		m_beat_generator.update(); // does some magic inside the beat_generator
			return {};
		}
    };

private:
  	// As is common practice, we provide private members at the
    // end of the class definition.
  	// Initializing values here still serves a purpose because this will occur
  	// prior to the constructor being called.
	std::unique_ptr<beat_generator>	m_beat_generator { nullptr };
};

```





