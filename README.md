torchx
======

This package contains various torch extensions:
 * [concat](#torch.concat) : concatenates a table of tensors.
 * [find](#torch.find) : finds all indices of a given value.
 * [group](#torch.group) : sorts and groups similar tensor variables together. 
 * [remap](#torch.remap) : recursively applies a function to tables of Tensors.
 * [md5](#torch.md5) : used for hashing strings.

And some [paths](https://github.com/torch/paths) extensions :
 * [indexdir](#paths.indexdir) : index a directory of millions of files for faster listing.


<a name='torch.concat'/>
### [res] torch.concat([res], tensors, [dim]) ###
Concatenates a table of Tensors along dimension `dim`.
 * `res` is a tensor holding the concatenation of Tensors `tensor`.
 * `tensors` is a table of tensors. Each tensor should have the same amount of dimensions and the same size for non-`dim` dimensions.
 * `dim` is the dimension along which the tensors will be concatenated. Defaults to 1.

Example:
```lua
> res = torch.concat({torch.rand(2,3),torch.randn(2,1),torch.randn(2,2)},2)
> print(res)
 0.8621  0.7776  0.3284 -1.2884 -0.4939  0.6049
 0.8404  0.8996  0.5704  0.3911 -0.0428 -1.4627
[torch.DoubleTensor of dimension 2x6]
```

<a name='torch.find'/>
### [res] torch.find(tensor, val, [dim]) ###
Finds all indices of a given value `val` in Tensor `tensor`. 
Returns a table of these indices by traversing the tensor one row 
at a time. When `dim=2`, the only valid value for dim other than `nil` (the default),
the function expects a matrix and returns the row-wise indices of each found 
value `val` in the row.

1D example:
```lua
> res = torch.find(torch.Tensor{1,2,3,1,1,2}, 1)
> unpack(res)
1  4  5
```

2D example:
```
> tensor = torch.Tensor{{1,2,3,4,5},{5,6,0.6,0,2}}
> unpack(torch.find(tensor, 2))
2	10	
> unpack(torch.find(tensor:t(), 2))
3	10	
> unpack(torch.find(tensor, 2, 2))
{2}  {5}
> unpack(torch.find(tensor:t(), 2, 2))
{ }  {1}  { }  { }  {2}
```

<a name='torch.group'/>
### [res, val, idx] torch.group([val, idx], tensor, [samegrp, desc]) ###
Sorts and groups similar tensor variables together.
 * `res` is a table of `{idx=torch.LongTensor,val=torch.Tensor}`.
 * `val` is a Tensor of the same type as `tensor`. It will be used to store and return the sorted values.
 * `idx` is a `torch.LongTensor` used to store the sorted indices.
 * `tensor` is a Tensor that will have its values sorted, and then grouped by the `samegrp` function.
 * `samegrp` is a function taking two argument : `first_val` is the first value of the current group, while `val` is the current value of the current group. When the function returns true, it is assumed that `val` is of the same group as `first_val`. Defaults to `function(first_val, val) return first_val == val; end`
 * `desc` is a boolean indicating whether the `tensor` gets sorted in descending order. Defaults to false.

Example:
```lua
> tensor = torch.Tensor{5,3,4,5,3,5}
> res, val, idx = torch.group(tensor)
> res
{
  3 : 
    {
      idx : LongTensor - size: 2
      val : DoubleTensor - size: 2
    }
  4 : 
    {
      idx : LongTensor - size: 1
      val : DoubleTensor - size: 1
    }
  5 : 
    {
      idx : LongTensor - size: 3
      val : DoubleTensor - size: 3
    }
}
```

<a name='torch.remap'/>
### [t1, t2] torch.remap(t1, t2, f(x,y) [p1, p2]) ###
Recursively applies function `f(x,y)` [to tables [of tables,...] of] Tensors
`t1` and `t2`. When prototypes `p1` or `p2` are provided, they are used 
to initialized any missing Tensors in `t1` or `t2`.

Example:
```lua
> t1 = {torch.randn(3,4), {torch.randn(3,4), torch.randn(2,4), {torch.randn(1)}}}
> t2 = {torch.randn(3,4), {torch.randn(3,4), torch.randn(2,4), {torch.randn(1)}}}
> torch.remap(t1, t2, function(x, y) x:add(y) end)
{
  1 : DoubleTensor - size: 3x4
  2 : 
    {
      1 : DoubleTensor - size: 3x4
      2 : DoubleTensor - size: 2x4
      3 : 
        {
          1 : DoubleTensor - size: 1
        }
    }
}
{
  1 : DoubleTensor - size: 3x4
  2 : 
    {
      1 : DoubleTensor - size: 3x4
      2 : DoubleTensor - size: 2x4
      3 : 
        {
          1 : DoubleTensor - size: 1
        }
    }
}
```
It also creates missing tensors:
```lua
> t2, t3 = torch.remap(t2, nil, function(x, y) y:resizeAs(x):copy(x) end)
> print(t3)
{
  1 : DoubleTensor - size: 3x4
  2 : 
    {
      1 : DoubleTensor - size: 3x4
      2 : DoubleTensor - size: 2x4
      3 : 
        {
          1 : DoubleTensor - size: 1
        }
    }
}
```
When in doubt, first tensor has priority:
```lua
> t4, t2 = torch.remap({torch.DoubleTensor()}, t2, function(x, y) x:resize(y:size()):copy(y) end, torch.LongTensor())
> print(t4)
{
  1 : DoubleTensor - size: 3x4
}
> t2, t5 = torch.remap(t2, {torch.DoubleTensor()}, function(x, y) y:resize(x:size()):copy(x) end, torch.LongTensor())
> print(t5)
{
  1 : DoubleTensor - size: 3x4
  2 : 
    {
      1 : LongTensor - size: 3x4
      2 : LongTensor - size: 2x4
      3 : 
        {
          1 : LongTensor - size: 1
        }
    }
}
```

<a name='torch.md5'/>
### torch.md5 ##

Pure Lua module copy-pasted from [this repo](https://github.com/kikito/md5.lua) (for some reasons I can't get 
git submodule to work with luarocks). The module includes two functions:
```lua
local md5_as_hex   = torch.md5.sumhexa(message)   -- returns a hex string
local md5_as_data  = torch.md5.sum(message)     -- returns raw bytes
```
The `torch.md5.sumhexa` function takes a string and returns another string:
```lua
torch.md5.sumhexa('helloworld!')
420e57b017066b44e05ea1577f6e2e12
```

<a name="paths.indexdir"/>
### [obj] paths.indexdir(path, [ext, use_cache, ignore]) ###
```lua
files = paths.indexdir("/path/to/files/", 'png', true)
images = {}
for i=1,files:size() do
   local img = image.load(files:filename(i))
   table.insert(images, img)
end
```

This function can be used to create an object indexing all files having 
extensions `ext` (a string or a list thereof) in directory `path` (string or list thereof). 
Useful for directories containing many thousands of files. The function 
caches the resulting list to disk in `/tmp` such that it can be used 
for later calls when `use_cache=true` (default is false). 
Argument `ignore` species a pattern to ignore (e.g. "*frame*" will ignore all files containing `"frame"`).
