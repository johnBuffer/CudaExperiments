# CudaExperiments

Test project for later cuda codes, currently only includes a mini wrapper around cuda calls to allow using exceptions instead of 

```cpp
status = cudaCall(...);
if (status != cuda_OK)
{
    // Zut an error
    ...
}
```

