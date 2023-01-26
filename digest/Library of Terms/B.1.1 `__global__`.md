## From the C++ Language Extensions Appendix
- Executed on the device
- Callable from the host
- Callable from the device for devices of compute capability 5.0 or higher
- Must have a `void` return type, and **cannot be amember of a class**.
- Any call to a `__global__` function must specify its execution configuration as described in [[B.34 Execution Configuration]]
- A call to a `__global__` function is **asynchronous**.