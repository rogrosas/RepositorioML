
def decide_label(prob: float, low: float = 0.30, high: float = 0.55) -> str:
    if prob >= high:
        return "RECHAZAR"
    elif prob < low:
        return "APROBAR"
    else: return "REVISIÓN MANUAL"
