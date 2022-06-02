import torch


def turn_off_debugging_processes(debug=True):
    # turn off pytorch debugging settings to speed up training
    if not debug:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.emit_nvtx(False)
        torch.autograd.profiler.profile(False)
