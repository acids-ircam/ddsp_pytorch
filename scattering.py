import torch
from kymatio.torch import Scattering1D
from torch.autograd import backward

# device = torch.device("cuda" if torch.cuda.is_available else "cpu")
device = "cpu"


def scattering_transform(onset_window, label, num_iterations):
    T = onset_window.shape[0]

    J = int(label + 1)
    Q = int(label + 1)

    splice = torch.tensor(onset_window)
    scattering = Scattering1D(J, T, Q).to(device)

    Sx = scattering(splice)

    learning_rate = 100
    bold_driver_accelerator = 1.1
    bold_driver_brake = 0.55

    # Random guess to initialize.
    torch.manual_seed(0)
    y = torch.randn((T, ), requires_grad=True, device=device)
    Sy = scattering(y)

    history = []
    signal_update = torch.zeros_like(splice, device=device)

    # Iterate to recontsruct random guess to be close to target.
    for k in range(num_iterations):
        # Backpropagation.
        err = torch.norm(Sx - Sy)

        print('Iteration %3d, loss %.2f' % (k, err.detach().cpu().numpy()))

        # Measure the new loss.
        history.append(err.detach().cpu())

        backward(err)

        delta_y = y.grad

        # Gradient descent
        with torch.no_grad():
            signal_update = -learning_rate * delta_y
            new_y = y + signal_update
        new_y.requires_grad = True

        # New forward propagation.
        Sy = scattering(new_y)

        if history[k] > history[k - 1]:
            learning_rate *= bold_driver_brake
        else:
            learning_rate *= bold_driver_accelerator
            y = new_y

    return y.detach().numpy()
