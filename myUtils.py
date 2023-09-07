def print_duration(duration):
    """
    Generates a formatted string representation of a given duration in hours, minutes, seconds, and milliseconds.

    Args:
        duration (float): The duration in seconds.

    Returns:
        str: The formatted string representation of the duration in the format "hours:minutes:seconds.milliseconds".
    """
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60 // 1
    miliseconds = abs((duration % 1) * 1000)
    return f"{int(hours)}:{int(minutes)}:{int(seconds)}.{int(miliseconds)}"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
