"""Lightweight hooks utility to capture intermediate model activations.

Usage:
    ma = ModelActivations(model, layers=["encoder.layer4.0.bn1"], detach=True)
    ma.register_hooks()
    ... forward pass ...
    acts = ma.fetch_activations("encoder.layer4.0.bn1")
"""


class ModelActivations:
    """Register forward hooks on a model to collect activations.

    Args:
        model: torch.nn.Module to attach hooks to.
        layers: list of layer names (module path strings) to capture.
        detach: if True, store detached CPU tensors to avoid retaining graph.
    """
    def __init__(self, model, layers, detach=False):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.handles = []
        self.detach = detach

    def _save_activation(self, name):
            def hook(module, inp, out):
                # store a detached tensor to avoid keeping computation graph
                try:
                    self.activations[name] = out.detach() if self.detach else out
                except Exception:
                    # out might be a tuple (e.g., some modules); try to detach each element
                    if isinstance(out, (list, tuple)):
                        # self.activations[name] = tuple(o.detach() if hasattr(o, 'detach') else o for o in out)
                        self.activations[name] = tuple(o for o in out)

                    else:
                        self.activations[name] = out
            return hook
    
    def register_hooks(self, layer=None):
        """Register forward hooks for the configured layers.

        If `layer` is provided, only that layer is registered.
        """
        for name, module in self.model.named_modules():
            if layer is not None:
                if name == layer:
                    if name not in self.layers:
                        self.layers.append(layer)
                    handle = module.register_forward_hook(self._save_activation(name))
                    self.handles.append(handle)
                    print(f"Registered hook for layer: {name}")
            else:
                if name in self.layers:
                    handle = module.register_forward_hook(self._save_activation(name))
                    self.handles.append(handle)
                    print(f"Registered hook for layer: {name}")

    def fetch_activations(self, layer_name):
        """Return captured activations for `layer_name` or None if absent."""
        return self.activations.get(layer_name, None)