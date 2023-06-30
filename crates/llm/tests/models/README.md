**Note**: These models are compressed, with most weights set to zero.
This makes them tiny. They still exercise the code, but the actual predictions
will be useless for production.

We ran this code to alter the models:

```python
    if layer_name.endswith("bias") or layer_name.endswith("weight"):
        assert len(layer.shape) > 0 and len(layer.shape) <= 3
        if len(layer.shape) == 1:
            layer[8:] = 0
        elif len(layer.shape) == 2:
            layer[:, 8:] = 0
        else:
            layer[:, :, 8:] = 0
```
