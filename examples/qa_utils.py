import gadgets


def apply_template(question: str, options: list[str]) -> str:
    if question.endswith("?"):
        return "%s Options: %s. " % (question, ", ".join(options))
    else:
        return "%s... Options: %s. " % (question, ", ".join(options))


def tagged_answer(answer_str: str) -> str:
    return "The answer is <%s>%s</%s>." % (gadgets.markup.RESULT_TAG, answer_str, gadgets.markup.RESULT_TAG)
