import importlib

def lazy_load_class(path: str):
    """
    Returns a function that imports and returns the class only when called.
    """
    module_name, class_name = path.split(":")
    
    def load():
        # Import the module dynamically
        module = importlib.import_module(module_name)
        # Get the class object from the module
        return getattr(module, class_name)
    
    return load