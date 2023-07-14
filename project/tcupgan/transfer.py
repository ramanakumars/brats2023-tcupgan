import torch
from torch.nn.parameter import Parameter


class Transferable():
    def __init__(self):
        super(Transferable, self).__init__()

    def load_transfer_data(self, checkpoint, verbose=False):
        state_dict = torch.load(checkpoint, map_location=next(self.parameters()).device)
        own_state = self.state_dict()
        state_names = list(own_state.keys())
        count = 0
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            # find the weight with the closest name to this
            sub_name = '.'.join(name.split('.')[-2:])
            own_state_name = [n for n in state_names if sub_name in n]
            if len(own_state_name) == 1:
                own_state_name = own_state_name[0]
            else:
                if verbose:
                    print(f'{name} not found')
                continue

            if param.shape == own_state[own_state_name].data.shape:
                own_state[own_state_name].copy_(param)
                count += 1

        if count == 0:
            print("WARNING: Could not transfer over any weights!")
        else:
            print(f"Loaded weights for {count} layers")
