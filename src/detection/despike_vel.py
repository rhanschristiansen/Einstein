def velocity_despike(y, window=1, threshold=10):
    """
    Args:
        y: data to despike
        window: window of velocity values to use during despike
        threshold: velocity threshold in window
    """
    y = y.copy()
    prev = y[0]
    for i in range(1, len(y)):
        abs_vel = abs(prev - y[i])
        if abs_vel > threshold:
            y[i] = prev
        prev = y[i]
    return y
