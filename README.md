# CudaExperiments

Test project for later cuda codes, currently only includes a mini wrapper around cuda calls to allow using exceptions. 

It turns this:
```cpp
status = cudaCall1(...);
if (status != cuda_OK)
{
    // Zut an error
    ...
}

status = cudaCall2(...);
if (status != cuda_OK)
{
    // Zut an error
    ...
}

status = cudaCall3(...);
if (status != cuda_OK)
{
    // Zut another error
    ...
}

// So much lines
```

Into this
```cpp
try
{
    cudaCallExcept1(...);
    cudaCallExcept2(...);
    cudaCallExcept3(...);
}
catch (const CudaException& ce)
{
    // Zut an error
    ...
}

// So clean

```
