torchx
======

This package contains various torch extensions.

<a name='torch.group'/>
### [res, val, idx] torch.group([val, idx], tensor, [samegrp, desc]) ###
Sorts and groups similar tensor variables.
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
Recursively applies function `f(x,y)` to table of Tensors (or Tensors)
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
> -- fills empty tensors
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
> -- tensor 1 has priority
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
> -- also works with tensors
> print(torch.remap(torch.randn(3), torch.randn(3), function(x, y) x:add(y) end))
-0.0486
-0.8466
-2.0608
[torch.DoubleTensor of dimension 3]

-0.0025
 0.2365
-1.4771
[torch.DoubleTensor of dimension 3]
```
