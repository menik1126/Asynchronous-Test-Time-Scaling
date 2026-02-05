from datasets import load_dataset
import random


def load_math500(repeat: int, test: bool):
    if test:
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        test_dataset = dataset["test"]
        random.seed(42)
        indices = random.sample(range(len(test_dataset)), 100)
        test_dataset = test_dataset.select(indices)
        dataset = test_dataset
    else:
        dataset = load_dataset("ricdomolm/MATH-500")
        train_dataset = dataset["train"]
        random.seed(43)
        indices = random.sample(range(len(train_dataset)), 200)
        train_dataset = train_dataset.select(indices)
        dataset = train_dataset

    problems = []
    answers = []
    for p, a in zip(dataset["problem"], dataset["answer"]):
        problems.extend([p] * repeat)
        answers.extend([a] * repeat)

    return problems, answers


def load_s1(repeat: int, test: bool):
    dataset = load_dataset("simplescaling/s1K-1.1")
    train_dataset = dataset["train"]
    random.seed(43)
    indices = random.sample(range(len(train_dataset)), 200)
    train_dataset = train_dataset.select(indices)
    dataset = train_dataset

    problems = []
    answers = []
    for p, a in zip(dataset["question"], dataset["solution"]):
        problems.extend([p] * repeat)
        answers.extend([a] * repeat)

    return problems, answers


def load_amc(repeat: int, test: bool):
    dataset = load_dataset("AI-MO/aimo-validation-amc")
    train_dataset = dataset["train"]
    indices = [i for i in range(50)]
    train_dataset = train_dataset.select(indices)
    dataset = train_dataset

    problems = []
    answers = []
    for p, a in zip(dataset["problem"], dataset["answer"]):
        problems.extend([p] * repeat)
        answers.extend([a] * repeat)

    return problems, answers


def load_aime24(repeat: int, test: bool):
    dataset = load_dataset("simplescaling/aime24_nofigures")
    train_dataset = dataset["train"]
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    train_dataset = train_dataset.select(indices)
    dataset = train_dataset

    problems = []
    answers = []
    for p, a in zip(dataset["problem"], dataset["answer"]):
        problems.extend([p] * repeat)
        answers.extend([a] * repeat)

    return problems, answers


def load_aime25(repeat: int, test: bool):
    dataset = load_dataset("simplescaling/aime25_nofigures")
    train_dataset = dataset["train"]
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    train_dataset = train_dataset.select(indices)
    dataset = train_dataset

    problems = []
    answers = []
    for p, a in zip(dataset["problem"], dataset["answer"]):
        problems.extend([p] * repeat)
        answers.extend([a] * repeat)

    return problems, answers


def load_gsm8k(repeat: int, test: bool):
    train_dataset = load_dataset("openai/gsm8k", "main")["train"]
    qa_pairs = list(zip(train_dataset["question"], train_dataset["answer"]))
    qa_pairs_sorted = sorted(qa_pairs, key=lambda x: len(x[1]), reverse=True)
    top_100 = qa_pairs_sorted[:100]
    top_100 = random.sample(qa_pairs, 100)
    dataset = top_100

    problems = []
    answers = []
    for item in dataset:
        problems.extend([item[0]] * repeat)
        ans = item[1]
        if "####" in ans:
            ans = ans.split("####")[-1].strip()
        else:
            exit(99)
        answers.extend([ans] * repeat)

    return problems, answers


def load_gpqa(repeat: int, test: bool):
    train_dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    indices = [i for i in range(60)]
    train_dataset = train_dataset.select(indices)

    questions = train_dataset["Question"]
    correct_answers = train_dataset["Correct Answer"]
    candidates_one = train_dataset["Incorrect Answer 1"]
    candidates_two = train_dataset["Incorrect Answer 2"]
    candidates_three = train_dataset["Incorrect Answer 3"]

    inputs = []
    labels = []
    for q, ca, c1, c2, c3 in zip(
        questions, correct_answers, candidates_one, candidates_two, candidates_three
    ):
        item = (q, ca, c1, c2, c3)
        inputs.extend([item] * repeat)
        labels.extend(["A"] * repeat)

    return inputs, labels


def load_olympiad(repeat: int, test: bool):
    dataset = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP")
    train_dataset = dataset["train"].filter(
        lambda example: example["is_multiple_answer"] is False
        and example["answer_type"] == "Numerical"
    )

    indices = [i for i in range(50)]
    train_dataset = train_dataset.select(indices)
    dataset = train_dataset
    problems = []
    answers = []
    for p, a in zip(dataset["question"], dataset["final_answer"]):
        problems.extend([p] * repeat)
        answers.extend(a * repeat)

    return problems, answers


def load_my_dataset(name: str, repeat: int = 1, test: bool = True):
    if name == "math500":
        return load_math500(repeat, test)
    if name == "s1":
        return load_s1(repeat, test)
    if name == "amc":
        return load_amc(repeat, test)
    if name == "aime24":
        return load_aime24(repeat, test)
    if name == "aime25":
        return load_aime25(repeat, test)
    if name == "gsm8k":
        return load_gsm8k(repeat, test)
    if name == "gpqa":
        return load_gpqa(repeat, test)
    if name == "olympiad":
        return load_olympiad(repeat, test)
