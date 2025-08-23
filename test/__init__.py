import importlib
import inspect
import pkgutil

# Iterate through the modules in the current package
for _, module_name, _ in pkgutil.walk_packages(__path__):
    # Import the module
    module = importlib.import_module(f"{__name__}.{module_name}")
    # Iterate through the attributes of the module
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        # Check if the attribute is a class
        if inspect.isclass(attribute):
            # Add the class to the package's namespace
            globals()[attribute_name] = attribute
