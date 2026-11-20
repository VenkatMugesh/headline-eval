from .schemas import Case, Grade

BANNED = ("guaranteed", "risk-free", "100%", "#1", "miracle")


def word_limit(case: Case, output: str) -> Grade:
    n = len(output.split())
    return Grade(criterion="word_limit", passed=n <= case.max_words, detail=f"{n} words")


def mentions_product(case: Case, output: str) -> Grade:
    ok = case.product.lower() in output.lower()
    return Grade(criterion="mentions_product", passed=ok)


def no_banned_words(case: Case, output: str) -> Grade:
    found = [w for w in BANNED if w in output.lower()]
    return Grade(criterion="no_banned_words", passed=not found, detail=", ".join(found))


def single_line(case: Case, output: str) -> Grade:
    ok = "\n" not in output and not output.lstrip().startswith(("-", "*", "1."))
    return Grade(criterion="single_line", passed=ok)


CODE_GRADERS = [word_limit, mentions_product, no_banned_words, single_line]
