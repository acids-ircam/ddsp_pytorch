# Min Guide to Threading in Max

Min objects are external objects for Max written in C++ using a high-level declarative application programming interface.  To get started refer to [Writing Min Objects](GuideToWritingObjects.md).

Additional information on Max's threading model can be found in the excellent *Computer Music Journal* article [Creating Visual Music in Jitter](https://www.scribd.com/document/86138715/Creating-Visual-Music-in-Jitter-Approaches-and-Techniques) by Randy Jones and Ben Nevile.

## The Max Threading Model

Max internally uses many threads for many purposes. All communication between objects in a patcher, however, occurs using a highly constrained subset of these threads:

1. The **Main** Thread — This is where most asynchronous computation and activity occurs in Max, including drawing UI objects, reading files, handling mouse-clicks, etc.

2. The **Scheduler** Thread — This thread is for timing sensitive computation. Examples are implementing a `metro` object or handling MIDI. It is not okay to do something that takes an indeterminant amount of time on this thread as it makes the timing performance of Max suffer.
   ​
   The scheduler "thread" is actually not always its own thread in the eyes of the operating system. If Max's **overdrive** setting is turned-off then the scheduler will be serviced in the main thread. If both Max's **overdrive** and **scheduler-in-audio-interrupt (SIAI)** settings are turned-on then the scheduler will be serviced by the audio thread. In all other cases the scheduler is actually its own thread.
   ​
   In light of this variability across different instances of the Max application it is important to be cautious about assumptions regarding thread-safety and re-entrancy in your objects.

3. The **Audio** Thread — This is where audio computation occurs. It is critical you **do not** perform operations that take an indeterminant amount of time on this thread. Most notably you should not allocate memory on the heap, lock a mutex, post to the console (internally it both allocates memory and locks a mutex), send messages out of outlets, etc.
   ​

   For more information on realtime-safety read Ross Bencina's [Time Waits for Nothing](http://www.rossbencina.com/code/real-time-audio-programming-101-time-waits-for-nothing) or watch Timur Doumler's [C++ in the Audio Industry](https://www.youtube.com/watch?v=boPEO2auJj4) talk from CppCon 2015.

Additional tasks or threads may be used within your objects, but all communication out of your object should be one of the aforementioned threads.


## Threading Considerations for Input

You will need to think about the thread handling of your object primarily at two stages of your object: input coming to your object, and preparing output that you send from your object.

As previously discussed, any given message to your object may come from Max's **Main** thread or from the **Scheduler** thread. When this happens your object may handle the situation in one of several ways:

1. Agnostically — paying no heed to the thread. This is appropriate for stateless operations such as calculating the square root of the input and outputting the result.
2. Forcing to the scheduler thread. To do this you trigger a `timer`. The operation defined for a timer will always run in the scheduler thread.
3. Deferring to the main thread. If the input is already on the main thread proceed, otherwise queue it for the next time the main thread is processed.

### Thread Safety

If input to your object occurs on multiple threads and your object's behaviour is dependent on some type of shared state (e.g. you have data storage that can be read by a message and set by another message or attribute) then you must take measures to ensure the validity of your object's state to guarantee a predictable result.

One option is to introduce locks. This can be simple or complex. The ramifications are often more complex than it appears at first glance.

Another option is to defer all operations to the main thread. This isn't always ideal. Consider the following patcher driven by a `metro`:

```
[metro]
 |
[counter]
 |\
 | \
 |  [acme.square]
 |   |
[+    ]
 |
```

If the number handling of the `acme.square` object is deferred then the output of the patcher will be *not what a user expects*. In this case the `+` object will process on the scheduler thread, triggered by the counter, and the input it receives from `acme.square` will almost certainly (though maybe occassionally not) be a previous output of the square because the operation has deferred and offloaded to a different thread.

In this case an "agnostic" approach seems ideal. But it isn't always ideal. Consider the scenario where `acme.square` is actually a working on a list that it maintains internally. An attribute determines the length of the list. While this patcher is running (in the scheduler thread) a user changes the length of the list in the object inspector. **Max crashes** as the scheduler accesses memory that is being reallocated. Or worse, memory is corrupted and some strange behavior happens in a random part of the patcher at an undetermined moment in the future. 

There are no easy answers.

### Thread Policy

The traditional Max API favors an agnostic approach to handling message input. 

The Min API favors a deferred approach. By default any `message<>` or `attribute<>` you create for your object in Min will be deferred unless you opt-in by saying a method is scheduler-safe. This can be done by specifying an optional template parameter ``message<threadsafe::yes>`. In attribute declarations the optional threadsafe parameter follows the underlying attribute type, for example `attribute<number, threadsafe::yes>`.

As we have seen there are good reasons for both approaches. The deferred approach can lead to unexpected behavior if not thought out. The consequences of the agnostic approach if not throught out, however, can be catastrophic and lead to program instability and unpredictability.

That said, you are not off the hook. *If you declare a `message<>` to be scheduler-safe you still must do the work to ensure that it really is scheduler safe*.


## Correct Threading for Output

It is essential that your object adhere to the Max threading model and send messages only in the main and scheduler threads. There are several tools available in Min to help your accomplish this.

1. The `timer` class. You can trigger a timer from any thread and it will run on the **scheduler** thread. Use a delay time of zero to trigger the timer to run immediately. It is safe to call outlets, post to the console, etc. from a timer. Please be a good citizen and don't bog down the scheduler with lengthy or indeterminant operations however.
2. The `queue` class. You can trigger a queue operation from any thread and it will run on the **main** thread as soon as possible.
3. The timer and queue will trigger an event (your function) to be called on a different thread. If you need data to be passed to the new thread as well then you can use a `fifo`. The FIFO is a "first-in, first-out" storage container for data of any type. You write into it from one thread and read from another. The fifo in Min is a lock-free implementation that is safe for use in the audio thread (or any other thread).

### An Example, Step-by-Step

As an example we could consider an object performing the same function as Max's built-in `edge~` external. This object looks at an audio signal to see if the input is zero or non-zero. When this changes it sends a bang in the scheduler thread out an outlet.

A naïve first implementation might look like this:

```c++
class edge : public object<edge>, sample_operator<1,0> {
public:

	inlet<>		input			{ this, "(signal) Input" };
	outlet<>	output_true		{ this, "(bang) input is non-zero" };
	outlet<>	output_false	{ this, "(bang) input is zero" };

	timer deliverer { this, 
        MIN_FUNCTION {
			if (state)
				output_true.send("bang");
			else
				output_false.send("bang");
			return {};
		}
    };

	void operator()(sample x) {
		if (x != 0.0 && prev == 0.0) { // change from zero to non-zero
			state = true;
			deliverer.delay(0);
		}
		else if (x == 0.0 && prev != 0.0) { // change from non-zero to zero
			state = false;
			deliverer.delay(0);
		}
		prev = x;
	}

private:
	sample	prev { 0.0 };
	bool	state { false };
};
```

If the audio sample input changes from zero to non-zero then switch our notion of state and trigger an output in the scheduler thread as soon as possible.

The problem with this implementation is that by time the scheduler is serviced the state may have changed. In this case the bang could come from the wrong outlet. Also, what happens if there is more than one zero / non-zero transition between servicings of the scheduler? Representing the data from this simple analysis of the signal as a single value is not adequate.

The solution is to use a FIFO buffer and information about each transition will be added to the FIFO. Then when the timer is serviced it will drain the FIFO buffer and deliver a bang for each transition that occurred.

```c++
class edge : public object<edge>, sample_operator<1,0> {
public:

	inlet<>		input			{ this, "(signal) Input" };
	outlet<>	output_true		{ this, "(bang) input is non-zero" };
	outlet<>	output_false	{ this, "(bang) input is zero" };

	timer deliverer { this, 
        MIN_FUNCTION {
			bool state;
			while (transitions.try_dequeue(state)) {
				if (state)
					output_true.send("bang");
				else
					output_false.send("bang");
			}
			return {};
		}
    };

	void operator()(sample x) {
		if (x != 0.0 && prev == 0.0) { // change from zero to non-zero
			transitions.try_enqueue(true);
			deliverer.delay(0);
		}
		else if (x == 0.0 && prev != 0.0) { // change from non-zero to zero
			transitions.try_enqueue(false);
			deliverer.delay(0);
		}
		prev = x;
	}

private:
	sample		prev { 0.0 };
	fifo<bool>	transitions { 100 };
};
```

The FIFO is initialized with an argument of `100` meaning that there will be space allocated for 100 `bool` values.

The audio routine calls `try_enqueue()` on the FIFO. This call will put the bool into the buffer if there is space available. If space is not available then nothing happens. There is an `enqueue()` that will allocate more memory if needed, but we want to avoid allocating memory in the audio thread so we have choosen a size for the FIFO at the outset that should meet our needs.

The timer function drains the FIFO using a while loop build on the `try_dequeue()` method.

Do we really want to deliver up to 100 bangs in a single scheduler tick? Maybe. Maybe not. Perhaps we want to format the output as a list of ones and zeros. Or make each of the two outlets output an `int` with the number of transitions. These strategies would thin the load placed on the scheduler thread to make the object more performant.

These same techniques also apply when using a `queue` instead of a `timer` if you want to move data to the main thread.

### High-Level Outlet Threading Specification

Rather than manually coding the timers, queues, and fifos to send from the audio thread, you can instead specify the threading behavior of your outlets. This will enforce delivery on a specific thread. 

If the outlet call is made on a thread other than the specified thread then an action will be performed. The action may be `assert` (crash before anything else can go wrong), `first` (output the first value received), `last` (output the last value received, aka "usurp"), or `fifo` (all values are queued as in our previous example).

The previous edge~ example could then be rewritten like this:

```c++
class edge : public object<edge>, sample_operator<1,0> {
public:
	inlet<> input { this, "(signal) input" };
  
	outlet<thread_check::scheduler, thread_action::fifo> output_true { 
      this, "(bang) input is non-zero" 
    };
  
	outlet<thread_check::scheduler, thread_action::fifo> output_false {
      this, "(bang) input is zero" 
    };

	void operator()(sample x) {
		if (x != 0.0 && prev == 0.0)
			output_true.send(k_sym_bang);	// change from zero to non-zero
		else if (x == 0.0 && prev != 0.0)
			output_false.send(k_sym_bang);	// change from non-zero to zero
		prev = x;
	}

private:
	sample prev { 0.0 };
};
```

In this case the manually queued version is more computationally efficient because no thread check is performed and the fifo size is fixed when the object is created. However, the declarative nature of the outlets makes this code clearer and less error-prone — and requires less typing.

## Using Locks

Writing scheduler-safe methods that are non-trivial (meaning dependent on state) requires thread-safety tools that may include both locks and lock-free techniques. 

When using locks there are some guidelines that you must consider:

1. Never call an outlet while a lock is held! This is an invitation to deadlocking and hangs.
2. Limit the scope of a lock to as small an area as is practical. Otherwise performance will suffer.



### Example: `min.list.process`

An example that uses locks is the `min.list.process` object in the Min-DevKit. It has the following private members representing the shared data and a mutex to protect the shared data:

```c++
private:
    atoms	m_data;
    mutex	m_mutex;
```

The function handling input contains a switch depending on the mode for which the object is operating.

```c++
		switch (operation) {
			case operations::collect: {
				lock lock { m_mutex };
				m_data.reserve( m_data.size() + args.size() );
				m_data.insert( m_data.end(), args.begin(), args.end() );
				break;
			}
			case operations::average: {
				lock lock { m_mutex };
				auto x = from_atoms<std::vector<double>>(args);
				auto y = math::mean<double>(x);
				lock.unlock();
				out1.send(y.first, y.second);
				break;
			}
			case operations::enum_count:
				break;
		}

```

In each case a lock is created using the mutex we create for our class. At this point we have exclusive access to the atoms in `m_data`. When the lock goes out of scope then the mutex is no longer active and others can access the shared data again. 

**Important Note #1:** Because we are relying on going out of scope we have wrapped the code for the case in `{ … }` braces.

**Important Note #2:** In the second case we manually unlock our mutex before making an outlet call — this is critical!

There is only one other place in this object where `m_data` is accessed, which is the "bang" message. 

```c++
message<threadsafe::yes> bang { this, "bang", "Send out the collected list.",
	MIN_FUNCTION {
		lock lock { m_mutex };
		atoms data_copy = m_data;
		m_data.clear();
		lock.unlock();
      
		out1.send(data_copy);
		return {};
	}
};
```

We must *not* call the outlet while `m_data` is locked. But `m_data` is the very thing we want to send to our outlet. The solution is to make a copy while the lock is held. Then unlock and send the copy to the outlet instead of the original.

## Example Projects

* `min.edge~` delivers output from the audio thread to the scheduler thread using the declarative outlet specification.
* `min.edgelow~` delivers output from the audio thread to the main thread using the declarative outlet specification.
* `min.sift~` delivers output to from the audio thread to either the scheduler or main thread depending on the setting of an attribute. The mechanism uses a manually configured timer, queue, and fifo.
* `min.list.process` uses locks to protect dynamically-sized shared memory for concurrent access by both the main and scheduler threads.
* `min.convolve` currently operates using the defaults — meaning everything is deferred to the main thread. It is an example that will require locks to be added before it can be declared thread-safe. This is left as an exercise for the diligent coder. When approaching the problem remember the order of operations for attribute setting — the attribute itself is not actually done until *after* you have returned from your setter function.

## Addendum: Attributes for Matrix Operators (Jitter)

Any objects you create that inherit from `matrix_operator` are **Jitter** classes. Under the hood there are some differences with regards to the thread-related behavior of attributes.

### Setters

A typical Min attribute setter defaults to a "usurp" behavior. This means that if the setter is called from a non-main thead that the execution of the setter will be deferred to the back of the queue. If multiple events occur before the queue is serviced then only the last value will be actually be set.

A Jitter attribute setter defaults to a "usurp low" behavior. This means that the call to the setter is *always* deferred to the back of the queue, even if it was called on the main thread. If multiple events occur before the queue is serviced then only the last value will be actually be set.

### Getters

At this point in time, getters for Min attributes occur synchronously — meaning that they are assumed to be threadsafe. Writing custom getters is atypical for most people coding externs, but if you do write a custom getter then please keep this in mind.

A Jitter attribute getter defaults to a "defer low". Meaning the call is *always* deferred to the back of the queue, even if called from the main thread. Additionally, it will be called once for every get "request" — not boiled down to a single call as in the "usurp" behavior of setters. 