/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include <atomic>

using namespace std::chrono_literals;

namespace c74 {
namespace mock {

    using function = std::function<void(void)>;
    using generic_callback = void(*)(void*); // TODO: convert to std::function?

    using clock_type = std::chrono::time_point<std::chrono::steady_clock>;
    using lock = std::lock_guard<std::mutex>;
    using timepoint = std::chrono::steady_clock::time_point;
    using duration_ms = std::chrono::milliseconds;

    class clock {

        class event {
            timepoint	m_onset;
//			function	m_meth;
            max::method	m_meth;
            void*		m_baton;

        public:
            event(timepoint onset, max::method meth, void* baton)
            : m_onset(onset)
            , m_meth(meth)
            , m_baton(baton)
            {}


            friend bool operator< (const event& lhs, const event& rhs) {
                return lhs.m_onset < rhs.m_onset;
            }

            friend bool operator< (const event& lhs, const timepoint rhs) {
                return lhs.m_onset < rhs;
            }

            friend bool operator > (const event& lhs, const timepoint rhs) {
                return lhs.m_onset > rhs;
            }

            void operator ()() const {
                m_meth(m_baton);
            }

            auto onset() const {
                return m_onset;
            }
        };


    public:
		clock(max::method a_callback, void* a_baton)
		: meth(a_callback)
		, baton(a_baton) {
			run_thread.test_and_set();
        }

        ~clock() {
			run_thread.clear();
			if (t.joinable()) {
				t.join();
            }
        }

        // TODO: mark class as not copyable?

        void unset() {
            {
                lock l(scheduled_events_mutex);
                scheduled_events.clear();
            }
            {
                lock l(pending_events_mutex);
                pending_events.clear();
            }
        }

        auto time(clock_type onset) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(onset - started_at).count();
        }

        auto now() {
            return std::chrono::high_resolution_clock::now();
        }

        void tick() {
			while (run_thread.test_and_set()) {
#ifdef WIN_VERSION
				// On Windows an exception will be thrown when we try to lock the mutex wihout some sort of time passage
				// (either a console post or an explicit short delay)
				// std::cout << c74::mock::clock::tick()" << std::endl;
				std::this_thread::sleep_for(1ms);
#endif
				std::unique_lock<std::mutex> pending_lock(pending_events_mutex);
				std::unique_lock<std::mutex> scheduled_lock(scheduled_events_mutex);

				// update list of events
				for (auto &e : pending_events)
					scheduled_events.insert(e);
				pending_events.clear();

				// User code can add new events, so don't lock pending_events
				// while calling the scheduled events!
				pending_lock.unlock();

				std::vector<event> events_that_ran;
				for (auto& e : scheduled_events) {
					if (e > now()) {
						//std::cout << time(now()) << " (all remaining events in the future)" << std::endl;
						break;
					}
					//std::cout << time(now()) << " (event to run)" << std::endl;
					events_that_ran.push_back(e);
					e();
				}

				// purge events that have run
				for (auto& e : events_that_ran)
					scheduled_events.erase(e);


				// events themselves may have added new (pending) events
				pending_lock.lock();
				for (auto &e : pending_events)
					scheduled_events.insert(e);
				pending_events.clear();
        
				const auto next = scheduled_events.empty() ? now()+1h : scheduled_events.begin()->onset();
				scheduled_lock.unlock(); // we're not touching any scheduled events anymore. Unlock the scheduled_events_mutex.
				cv.wait_until(pending_lock, next);
			}
        }

        void add(const std::chrono::milliseconds delay/*, function meth*/) {
            auto			onset = now() + delay;
            clock::event	event {onset, meth, baton};

            //std::cout << time(now()) << " NEW EVENT: " << std::chrono::duration_cast<std::chrono::milliseconds>(onset - started_at).count() << "   " << baton << std::endl;

            // a previous version of this code incorrectly tried to deal with deadlock by checking to see if the tick thread was the one adding a new event.
            // but if the tick thread was adding an event simultaneously to another thread then the content of new_events would be undefined, so we need to lock it for all writing.
            // the deadlock was occurring because the tick() method was calling into user code while locked, and the user code (in the metro case) was adding new events.

            lock l(pending_events_mutex);
            pending_events.push_back(event);
            cv.notify_one();
        }

        /** Convenience wrapper for adding events with a callback to a class member method without having to manually setup std::bind.
            @param delay	time offset from now() at which to execute
            @param meth		pointer to member method to be called
            @param target	pointer to this
         */
//		template<class MethType, class T>
//		void add(const std::chrono::milliseconds delay, MethType meth, T* target) {
//			add(delay, std::bind(meth, target));
//		}


        const std::thread::id threadid() const {
            return t.get_id();
        }

    private:
        clock_type				started_at = { std::chrono::steady_clock::now() };
        std::thread				t { &clock::tick, this };
        std::mutex				pending_events_mutex; // mutex that must be locked when accessing pending_events
        std::vector<event>		pending_events;
        std::mutex				scheduled_events_mutex; // mutex that must be locked when accessing scheduled_events
        std::multiset<event>	scheduled_events;	// using multiset instead of vector because we want to keep it sorted, performance impacts?
        std::condition_variable	cv;
		std::atomic_flag        run_thread;

        max::method				meth;
        void*					baton;
    };

}} //namespace c74::mock


namespace c74 {
namespace max {

    using t_clock = mock::clock;
    using t_timeobject = t_object;


    MOCK_EXPORT t_clock* clock_new(void* obj, method fn) {

        // TODO: we're going to run into trouble freeing this with object_free...

        return new mock::clock(fn, obj);
    }


    MOCK_EXPORT void clock_unset(t_clock* self) {
        // TODO: implement
		self->unset();
    }


    MOCK_EXPORT void clock_fdelay(t_clock* self, double duration_in_ms) {
        self->add(std::chrono::milliseconds((int)duration_in_ms));
    }


    MOCK_EXPORT	double time_getms(t_timeobject *x) {
        return 0;
    }


}} // namespace c74::max
