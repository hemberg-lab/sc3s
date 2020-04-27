# these will be accessible directly as `example.addx`
# and the long form `example.operations.addx`
from .operations import addx, subx

"""
if you remove the above line, then in a python terminal:

    from example.operations import addx, subx, multx, divx

The functions become accessible as they are named (not recommended IMO)
Correction: this actually doesn't work!

Which makes sense, because the syntax I use to call it would be
example.operations.addx

This is similar as a subdirectory method.

`import operations` does not work.

BUT `from . import operations as op` does work!
Functions can then be accessed as: `example.op.addx`, but not `addx`
"""

##########

# here is a subdirectory within the module folder: accessed as example.sd.multx
from . import subdirectory as sd

"""
`import subdirectory` does not work.

But, as above, `from .subdirectory import multx, divx` does work.
Functions can then be accessible as succinitly or `example.multx`, 
or the long form `example.subdirectory.multx`.

#####

Note: inside the subdirectory's `__init__.py` file, must have this
`from .operations import multx, divx`

`import multx, divx` does not work.

However, if these functions are in the `__init__.py` file,
then they can work, even without needing any extra `import` statements.
"""

##########

"""
My takeaway from this is that the `import` function seems to treat subdirectories 
and scripts the same. The subdirectory method is nice when you have too many functions
to fit in a single script.
"""