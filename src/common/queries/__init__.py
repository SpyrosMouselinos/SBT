import inspect
import importlib

module_name = 'queries'
module = importlib.import_module(f'.{module_name}', package=__package__)

# Filter functions that start with 'get'
get_functions = [name for name, obj in inspect.getmembers(module, inspect.isfunction) if name.startswith('get')]
for func_name in get_functions:
    globals()[func_name] = getattr(module, func_name)

# Update __all__ with the list of 'get' functions
__all__ = get_functions
