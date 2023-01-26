## Definition
Any call to a [[B.1.1 `__global__`]] function must specify the *execution configuration* for that call.
- Defines the dimension of the grid and blocks that will be used to execute the function on the device, as well as the associated stream.
- The configuration is specified by inserting an expression of the form:
	- `<<< Dg, Db, Ns, S >>>`
	- `Dg` is of type `dim3` (see [[B.3.2 dim3]]) and specifies the dimension and size of the **grid**, such that `Dg.x * Dg.y * Dg.z` equals **the number of blocks being launched**;
	- `Db` is of type `dim3` (see [[B.3.2 dim3]]) and specifies the dimension and size of **each block**, such that `Db.x * Db.y * Db.z` equals **the number of threads per block**.
	- `Ns` is of type `size_t` and specifies the number of bytes in **shared memory** that is **dynamically allocated per block _for this call_** in addition to the statically allocated memory.
		- This dynamically allocated memory is used by **any of the variables declared** as an external array as mentioned in [[B.2.3 __shared__]]. `Ns` is an **optional** argument which defaults to 0.
	- `s` is of type `cudaSteream_t` and specifies the **associated stream**; it is optional and defaults to 0.
	
## Sample usage
A function declared as
```cpp
__global__ void Func(float* parameter);
```
has to be called like this
```cpp
Func<<< Dg, Db, Ns>>>(parameter);
```
