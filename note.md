Normally PyTorch would look something like this

```py
x = nn.Linear(in_sz, out_sz)(inp)
x = nn.Linear(in_sz, out_sz)(x)
x = nn.Linear(in_sz, out_sz)(x)
x = nn.Linear(in_sz, out_sz)(x)
```