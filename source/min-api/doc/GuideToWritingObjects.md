# Writing Min Objects

Min objects are external objects for Max written in C++ using a high-level declarative application programming interface.

Example code is distributed as a part of the [Min-DevKit Package](https://github.com/Cycling74/min-devkit).

See Also:

* [Guide To Threading](GuideToThreading.md)
* [Guide To Audio](GuideToAudio.md)
* [Guide To Initialization](GuideToInitialization.md)
* [Guide To UI Objects](GuideToUserInterfaceObjects.md)
* [Special Messages](SpecialMethods.md)
* [Where To Look...](WhereToLook.md)


## Includes

You will need to include one header file to write a Min object.

```c++
#include "c74_min.h"
```

## Class Definition

To create a Min object you define a class that inherits from a specialization of the `min::object` class. You then wrap this class with the `MIN_EXTERNAL` macro that exposes the class to Max.

```c++
class my_object : public object<my_object> {
public:
	/// ...
};

MIN_EXTERNAL(my_object);
```

Note that the `object` which you are extending is itself specialized with the type of your class. This idiom provides a means of achieving [static polymorphism](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern#Static_polymorphism) which is important for efficiency in Min.


## Constructors

You may provide a constructor for your object if you have some custom initialization to perform. If you do, it should possess a `const atoms&` parameter for passing in arguments typed into the Max object box.  

The parameter will be a vector of all atoms entered as arguments that occur prior to any attribute arguments.  Thus `my_object foo bar @thing 2` will pass a vector of size=2 with the contents being an atom each for `foo` and `bar`.

For example, a filter object might have a constructor that looks like this:

```c++
my_object(const atoms& args = {}) {
	if (args.size() > 0)
		frequency = args[0];
	if (args.size() > 1)
		resonance = args[1];
	calculate_coefficients();
}
```

Attributes are created and defaults assigned prior to the constructor being called.
Attribute values entered into the Max object box are processed after the constructor has finished.

If you need to abort construction of your object, call `error()`. This will throw an exception which will be caught by the caller trying to construct your object and the associated object box in the patcher will then be disabled.


## Destructors

If you need to do any tear-down when your object is freed then simply define a destructor:

```c++
~my_object {
	// object-specific tear-down code here
}
```

## Inlets and Outlets

The first thing you will want to define for your new class are inlets and outlets. To do so you create an instance of the `inlet<>` or `outlet<>` type and initialize it with a pointer to your class' instance (`this`) and an assistance string for users that describes what the inlet or outlet does.

```c++
class my_object : public object<my_object> {
public:
	inlet<>		input	{ this, "(list) values to convolve" };
	outlet<>	output	{ this, "(list) result of convolution" };

	/// ...
```

Inlets and outlets may be generic, as above, or they may be specific to a type. Below, the left inlet is generic but the right inlet and the outlet both have the optional type defined for "dictionary". Audio objects will typically have outlets defined with the "signal" type and Jitter objects will typically use the "matrix" type.

```c++
inlet<>		left	{ this, "dict to combine with dict at right inlet" };
inlet<>		right	{ this, "dict to combine with dict at left inlet", "dictionary" };
outlet<>	output	{ this, "dictionary of entries combined from both inlets", "dictionary" };
```

Both inlets and outlets are defined in left to right order (for users of the traditional Max-SDK in C this is the opposite of what you may be used to).

### Inlet Types

Inlet "types" are actually the name of the message which is received by your object in Max. Integer messages have an invisible message name `int`, floats an invisible message name `float` and lists starting with a number have an invible message name `list`.  All other messages are the name of the symbol that begins the message. For example, if you send a "bang" message to your object the message type is "bang".

In most cases you will likely use generic inlets and outlets that accept (and produce) any type. Some common types are:

* int
* float
* list
* bang
* signal
* dictionary
* matrix

### Configuring Inlets and Outlets at Runtime

In most cases configuring inlets and outlets at compile time is the ideal solution. There are cases, however, where you may wish to define the inlets and outlets at runtime based on the arguments passed to your object's constructor. 

(note: you cannot define the number of inlets and outlets at runtime for classes inheriting from `sample_operator<>` , you must instead inherit from `perform_operator<>` as is done in the *filter.dcblocker~* example code in the Filter package.)

In an example where you wish to define both the inlets and the outlets at runtime, you will need to create a place in your class to store the inlet/outlet instances. A convenient way to store the instances is in a vector.

```c++
private:
    std::vector< std::unique_ptr<inlet<>> >	m_inlets;
    std::vector< std::unique_ptr<outlet<>> >	m_outlets;
```

In your constructor you then create the inlets and outlets and add the instances to your vectors. Here is an example that takes the value of an argument to determine the number of inlets and outlets.

```c++
/// constructor
clone(const atoms& args = {}) {
  if (args.empty())
    error("argument required");

  auto inlet_count = args[0];
  auto outlet_count = inlet_count * 2;
  
	for (auto i=0; i < inlet_count; ++i) {
    auto an_inlet = std::make_unique<inlet<>>(this, "(bang) my assist message");
    m_inlets.push_back( std::move(an_inlet) );
  }

  for (auto i=0; i < outlet_count; ++i) {
    auto an_outlet = std::make_unique<outlet<>>(this, "my outlet assist message");
    m_outlets.push_back( std::move(an_outlet) );
  }
}
```

## Messages

The basic work of most Max objects is done by messages. All messages take a single `const atoms&` parameter just as constructors. If you don't need arguments for your message then you can simply ignore it as in this example message:

```c++
message<> bang { this, "bang", "Post something to the Max console.",
	MIN_FUNCTION {
		cout << "Hello World" << endl;
		return {};
	}
};
```
When you define a message you are creating an instance of a `min::message<>` and initializing it with a pointer to the owning instance of your class (`this`), a string for the message name in Max, a description string for documentation, and a function to be executed when the message is called. Typically the function is defined using a C++ lambda, whose verbose signature is tucked-away in the `MIN_FUNCTION` macro. 

(A lambda is an anonymous callback function that is triggered in response to an event — a message being received by your object. You can also define a function and then pass that function to the constructor. The *min.threadcheck* example code in the Min-Devkit uses this latter method.)

The signature of `MIN_FUNCTION` says that it will take `const atoms&` as input and return `atoms` as output.  Most messages won't have a return value so you can just return an empty set of atoms as in the example above.

If you wish to access the arguments to your message, do so the same way as described for the constructor as in the example that follows. Any `MIN_FUNCTION` can access it's arguments as a vector of atoms named `args`.

```c++
message<> number { this, "number", 
    MIN_FUNCTION {
		position = args[0];
		return {};
	}
};
```
A "number" message will be called for either "float" or "int" input. If you want to only handle ints then define an "int" message; if you want to only handle floats then define a "float" message.


## Attributes

Attributes are simply variables that are exposed to Max. To do this you create attribute instance specialized with the datatype the attribute is to represent.

Attributes have 3 required arguments: a pointer to the owning instance of your class (`this`), a string for the attribute name in Max, and a default value for initialization.

```c++
attribute<double> min { this, "minimum", 0.0 };
attribute<double> max { this, "maximum", 1.0 };
```

Following the 3 required arguments, attributes may have any number of optional arguments, which may be in any order:

* `title`: this is a human-friendly label for your attribute shown in the inspector
* `description`: a documention string describing the attribute
* `range`: for numerical attributes this will be two values representing the low and high limits of the number; for symbols this will be a list of possible options available to be specified
* `setter`: a function to be run prior to assigning the value
* `getter`: a custom function for fetching the stored value
* `readonly` : a bool that indicates an attribute is not user-writable

An attribute that uses just the `setter` might look like this:

```c++
attribute<bool> on { this, "on", false,
	setter { MIN_FUNCTION {
		if (args[0] == true)
			metro.delay(0.0);	// fire the first one straight-away
		else
			metro.stop();
		return args;
	}}
};
```

And an example that uses multiple of these optional arguments might look like this:

```c++
attribute<symbol> mode {
	this,
	"mode",
	"fast",
	setter { MIN_FUNCTION {
		std::tie(weight1, weight2) = calculate_weights(args[0], position);
		return args;
	}},
	title {"Calculation Modality"},
	range {"fast", "precision"}
};
```

#### The Value Attribute

If you name your attribute "value" it will be imbued with additional status: this attribute will represent your object to Max's preset, pattr, and parameter (snapshot) systems for saving and recalling patcher state.

### Range Limiting

Providing a range for the `bool` or  `enum` types automatically limits input to the options available. For other numeric types the provided range is a suggestion and is used to generate documentation.

```c++
attribute<number> foo { this, "foo",0.5,
	range { 0.0, 1.0 }
};
```

To enforce the range to be limited you can specialize the attribute with a `limit` type.

```c++
attribute<number, threadsafe::no, limit::clamp> foo { this, "foo",0.5,
	range { 0.0, 1.0 }
};
```

In this example values lower than zero will be "clamped" to zero and values greater than one will be clamped to one because the attribute is specialized with the `limit::clamp` parameter. 

In order to specialize the limit type you also must specify the threadsafety of the attribute. By default this is `threadsafe::no` and you should choose `threadsafe::no` unless you have thoroughly read the [Guide To Theading](GuideToThreading.md) and you are confident that you are making the correct choice.

The options for limiting are

* `limit::none`: This is the default
* `limit::clamp`: Values below the range are held at the bottom of the range, values above are held at the top of the range.
* `limit::wrap`: Values that go out of range at the top wrap around to the bottom and keep increasing. Values that go out of range at the bottom wrap around to the top and keep decreasing.
* `limit::fold` : Values that go out of range at the top start mirroring back down into the range from the top. If they then further exceed the bottom of the range they will fold again back up into the range. Values that go out of range at the bottom follow the same pattern.

Boundary limit behaviours are applied prior to any custom setters being called.

### Repetition Filtering

When input to your object repeatedly sets the attribute to the same value it can consume computational resources or result in other undesired behavior. It is possible to filter out such repeated setting of the same value using an optional fourth template argument.

```c++
attribute<number, threadsafe::no, limit::clamp, allow_repetitions::no> foo { this, "foo",0.5,
	range { 0.0, 1.0 }
};
```

By default repetitions are allowed. 

### Custom Setters

Custom setters use the same `MIN_FUNCTION` signature as messages above. This means it will take `const atoms&` as input and return `atoms` as output.  The input will be the value coming from the patcher and the value that is returned is what will be assigned to the attribute.

Often, as in the examples above, the setter is used to produce a side effect. Another use of custom setters is to check the input for validity prior to assignment and make alterations if neccessary.

### Vector Attributes

Array/Vector attributes are defined by using a specialization of `std::vector` for the attribute type. Here is an example from the **min.convolve** object in the Min-DevKit.

```c++
attribute< vector<double> > kernel { this, "kernel", {1.0, 0.0} };
```

Note that the initialization of the attribute must be wrapped in curly braces.


## Posting to the Console

To post to the Max console use `cout` (normal messages) and `cerr` (error messages).  Your message will not post until `endl` is received by the stream.

```c++
method anything { this, "anything", 
    MIN_FUNCTION {
		cout << "Message Received: " << args << " !" << endl;
		// ...
		return {};
	}
};
```

To post to the system console instead of the Max console use `std::cout`, `std::cerr`, and `std::endl` as is typical in C++ rather than the variants implemented in the `c74::min` namespace.

## Timers

To schedule an event to happen at some point in the future use a `min::timer`. Timers use a pattern that hopefully is becoming familiar: you create an instance of timer and initialize it with a pointer to an instance of your class (`this`) and function (typically a lambda function) that will be executed when the timer fires.

```c++
timer metro { this, 
    MIN_FUNCTION {
		bang_out.send("bang");
		metro.delay(interval);
		return {};
	}
};
```

In the example above `bang_out` is an outlet. After sending the "bang" the timer schedules itself to run again at an interval in milliseconds.

## Queues

A `min::queue` creates an element that, when set, will be executed by Max's low-priority queue. This provides a mechanism for transferring or deferring events from other threads (such as the scheduler or audio thread) to Max's main thread.

```c++
queue deferrer { this, 
    MIN_FUNCTION {		
		bang_out.send("bang");
		return {};
	}
};

// elsewhere in your code call this to fire the queue element:
// deferrer.set()
```


## Text Editor Windows

To add a text editor window to your object simply add a `texteditor` instance to your class. You will initialize it with a pointer to an instance of your class (`this`) and a lambda that will be called when the editor window is closed. 

Note that the lambda is *not* a `MIN_FUNCTION` but rather a special lambda that passes in the text content of the editor window.

```c++
texteditor editor { this, 
    [this](const char* text) {
		// do something with the text...
		// e.g. save it in a member variable, turn it into atoms, etc.
	}
};
```


## Dictionaries

Dictionaries are Max's implementation of an associative array container mapping symbols (keys) to data. There are a variety of ways you might use dictionaries.

### Handling Dictionary Input

To respond to a dictionary coming into an inlet, define a message named "dictionary". It' first argument will be an atom containing a dictionary.

``` c++
message<> dictionary { this, "dictionary", 
    MIN_FUNCTION {
		dict d { args[0] };
		sequence = d["pattern"];
		return {};
	}
};
```
In this example the dict "d" is constructed using an atom containing a dictionary. It is important to understand that this dictionary is *not* a copy of the dictionary, but rather a reference. As long as "d" is in scope the reference will be valid.

Next, a variable named "sequence" is assigned a value from the dictionary that is stored with the key name "pattern". 

If "pattern" doesn't exist it will be created and sequence will be assigned an empty set of atoms. If you wish to use bounds checking and have an error thrown then use the `at()` method of `dict` instead of the `[]` operator as in the following example:

```c++
message<> dictionary { this, "dictionary", 
    MIN_FUNCTION {
		dict d { args[0] };
		try {
			sequence = d.at("pattern");
		}
		catch (std::runtime_error& e) {
			cerr << "could not fetch key called 'pattern'" << endl;
		}
		return {};
	}
};
```

## Saving State

Most state saving in Max is handled automatically via the attribute system. If you need to save additional custom state define a 'savestate' message. This message will receive an atom containing a dictionary as input. Write your data into this dictionary to have it saved with the patcher

```c++
message<> savestate { this, "savestate", 
    MIN_FUNCTION {
		dict d { args[0] };
		d["my_custom_data"] = some_data;
		return {};
	}
};
```

To recall your saved state when the patcher is loaded, the object is pasted into another patcher, etc. you call the inherited `state()` method to get your instance's dictionary from the patcher.

```c++
auto saved_state = state();						
auto some_data = saved_state["my_custom_data"];
if (some_data.empty()) 
	; // no atoms were returned...
else
	; // atoms were returned so do something with them
```

## Custom Max Class and Instance Callbacks

In some cases you may wish to do some advanced class setup. The example below could (and should) be done with optional parameters to the attribute, but it demonstrates how the mechanism works.

```c++
// the "maxclass_setup" method is called when the class is created
// it is not called on an instance at what we think of in Max as "runtime"
message<> maxclass_setup { this, "maxclass_setup", 
    MIN_FUNCTION {
		c74::max::t_class* c = args[0];

		CLASS_ATTR_ENUM(c,	"shape", 0, "linear equal_power square_root");
		CLASS_ATTR_LABEL(c,	"shape", 0, "Shape of the crossfade function");

		return {};
	}
};
```

## Underlying Details About Class Creation

The process by which a class is registered with the Max kernel involves the creation of a "dummy" instance of your object the first time the object is requested. This dummy instance is then interrogated and it's various properties are bound to equivalent structures in Max. The dummy instance is then destroyed and all subsequent instantiations of your object are "real" instantiations.

If you need to know when you are working with a dummy instance, then you can call the `dummy()` function. This will return true if the instance is a dummy.



## Unit Testing

Unit testing is performed using Catch framework. See the ReadMe for more details.

