CXX = g++

# Change this for your system
PYTHON_LIBS = $(shell pkg-config --libs python3)
PYTHON_CFLAGS = $(shell pkg-config --cflags python3)
TENSORFLOW_CFLAGS = $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TENSORFLOW_LIBS = $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CXXFLAGS = -O3 -g -Wall -Werror -std=c++11 -Wno-unknown-pragmas

ALLTARGETS = main_policy test_policy count_characters.so expand_operation.so


all: $(ALLTARGETS)

main_policy: policy.cc main_policy.cc policy.h
	$(CXX) $(CXXFLAGS) -o $@ policy.cc main_policy.cc

test_policy: policy.cc test_policy.cc policy.h
	$(CXX) $(CXXFLAGS) -o $@ policy.cc test_policy.cc

count_characters.so: policy.h policy.cc count_characters.cc python_counter.h python_counter.cc
	$(CXX) $(CXXFLAGS) -fPIC $(PYTHON_LIBS) $(PYTHON_CFLAGS) -o $@ -shared policy.cc count_characters.cc python_counter.cc

expand_operation.so: policy.h policy.cc expand_operation.cc
	$(CXX) $(CXXFLAGS) -fPIC -shared $(TENSORFLOW_CFLAGS) -Wl,--no-as-needed $(TENSORFLOW_LIBS) -Wl,--as-needed -o $@ policy.cc expand_operation.cc

clean:
	rm $(ALLTARGETS)
