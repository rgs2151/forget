BASELINE_SYSTEM = "You are knowledgeable and answer truthfully. Answer concisely in one phrase."


def refuse_system(concept):
    return (
        f"Pretend you have no knowledge of {concept}. "
        f"You cannot recall any information about {concept}. "
        f"If asked, say 'I don't know.'"
    )
