from typing import Optional
from parsing import ParsedCase


SYSTEM_PROMPT = (
    "You are a judicial reasoning assistant trained to analyze criminal appeal "
    "cases the way an Indian appellate court would. You are given the unified "
    "facts, key actors, timeline, claims, cited precedent, and the statutory "
    "provisions relevant to a case. Working strictly from the material "
    "provided, you reason like both sides of the bar and then like the bench: "
    "you build the strongest case FOR the prosecution's theory, the strongest "
    "case AGAINST it (on behalf of the defense/appellant), weigh which "
    "arguments withstand scrutiny and which collapse, and finally deliver a "
    "reasoned holding and result. Ground every point in a specific fact, "
    "timeline entry, statutory provision, or precedent from the material "
    "given -- never invent facts, evidence, or authorities that are not "
    "implied by it."
)

GROUNDING_REMINDER = (
    "Ground every point in a specific fact, timeline entry, statutory "
    "provision, or precedent from the material above -- do not invent facts "
    "or authorities. Respond with only the requested reasoning, not a "
    "restatement of these instructions."
)


# --------------------------------------------------------------------------- #
# Context block builder (shared "facts + law" input used by every task)
# --------------------------------------------------------------------------- #

def build_context_block(pc: ParsedCase) -> str:
    return (
        f"## Case Facts\n{pc.facts}\n\n"
        f"## Relevant Sections of Law\n{pc.law}"
    )


def build_arguments_block(pc: ParsedCase) -> str:
    return (
        f"### Arguments FOR (Prosecution)\n{pc.arguments_for}\n\n"
        f"### Arguments AGAINST (Defense)\n{pc.arguments_against}"
    )


def build_evaluation_block(pc: ParsedCase) -> str:
    return (
        f"### Reasoning to Accept Arguments\n{pc.acceptance_logic}\n\n"
        f"### Reasoning to Reject Arguments\n{pc.rejection_logic}"
    )


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _example(case_id: str, task: str, messages: list, extra: Optional[dict] = None) -> dict:
    row = {
        "id": f"{case_id}__{task}",
        "case_id": case_id,
        "task": task,
        "messages": messages,
    }
    if extra:
        row["meta"] = extra
    return row


# --------------------------------------------------------------------------- #
# Task A -- Argument generation (FOR + AGAINST)
# --------------------------------------------------------------------------- #

def build_task_arguments_generation(pc: ParsedCase) -> dict:
    context = build_context_block(pc)
    user = (
        "Below are the unified facts, key actors, evidentiary timeline, "
        "claims, cited precedent, and the statutory provisions relevant to a "
        "criminal appeal.\n\n"
        f"{context}\n\n"
        "Using only this material, construct:\n"
        "1. The strongest arguments FOR the prosecution's case (i.e., "
        "arguments supporting conviction).\n"
        "2. The strongest arguments AGAINST the prosecution's case (i.e., "
        "arguments for the defense/appellant favoring acquittal or a lesser "
        "offense).\n\n"
        f"{GROUNDING_REMINDER} Present the two sets as clearly separated, "
        "numbered lists under the headings \"Arguments FOR (Prosecution)\" "
        "and \"Arguments AGAINST (Defense)\"."
    )
    assistant = build_arguments_block(pc)
    messages = [_msg("system", SYSTEM_PROMPT), _msg("user", user), _msg("assistant", assistant)]
    return _example(pc.case_id, "arguments_generation", messages)


# --------------------------------------------------------------------------- #
# Task B -- Argument evaluation (accept / reject)
# --------------------------------------------------------------------------- #

def build_task_argument_evaluation(pc: ParsedCase) -> dict:
    context = build_context_block(pc)
    args = build_arguments_block(pc)
    user = (
        "Here is the case material, followed by the arguments already "
        "constructed for and against the prosecution's case.\n\n"
        f"{context}\n\n"
        f"{args}\n\n"
        "Now evaluate these arguments as an appellate court would. For each "
        "side, determine which specific arguments are persuasive enough to "
        "accept and which must be rejected, and explain the legal reasoning "
        "behind each determination (e.g., sufficiency of evidence, "
        "completeness of the chain of circumstantial evidence, credibility "
        "of witnesses, applicable statutory presumptions and whether they "
        "have been rebutted). Separate your analysis into \"Reasoning to "
        f"Accept Arguments\" and \"Reasoning to Reject Arguments\". {GROUNDING_REMINDER}"
    )
    assistant = build_evaluation_block(pc)
    messages = [_msg("system", SYSTEM_PROMPT), _msg("user", user), _msg("assistant", assistant)]
    return _example(pc.case_id, "argument_evaluation", messages)


# --------------------------------------------------------------------------- #
# Task C -- Final judgment
# --------------------------------------------------------------------------- #

def build_task_final_judgment(pc: ParsedCase) -> dict:
    context = build_context_block(pc)
    args = build_arguments_block(pc)
    evaln = build_evaluation_block(pc)
    user = (
        "Here is the full case material, the arguments for and against the "
        "prosecution, and the court's reasoning on which arguments to accept "
        "and which to reject.\n\n"
        f"{context}\n\n"
        f"{args}\n\n"
        f"{evaln}\n\n"
        "Based on all of the above, deliver the court's final holding. State "
        "clearly whether the prosecution's case is upheld or fails, explain "
        "the decisive reasoning, and give the concrete result/order (e.g., "
        "conviction affirmed or set aside, sentence, acquittal, remand). "
        f"{GROUNDING_REMINDER}"
    )
    assistant = pc.final_conclusion
    messages = [_msg("system", SYSTEM_PROMPT), _msg("user", user), _msg("assistant", assistant)]
    return _example(pc.case_id, "final_judgment", messages)


# --------------------------------------------------------------------------- #
# Task D -- Full chain, single turn (one completion, four labeled stages)
# --------------------------------------------------------------------------- #

def build_task_full_chain_single_turn(pc: ParsedCase) -> dict:
    context = build_context_block(pc)
    user = (
        "Below are the unified facts, key actors, evidentiary timeline, "
        "claims, cited precedent, and statutory provisions relevant to a "
        "criminal appeal.\n\n"
        f"{context}\n\n"
        "Working only from this material, produce a complete judicial "
        "reasoning trace with the following four stages, clearly labeled and "
        "in order:\n\n"
        "1. **Arguments FOR (Prosecution)** -- the strongest case for "
        "conviction.\n"
        "2. **Arguments AGAINST (Defense)** -- the strongest case against "
        "conviction.\n"
        "3. **Evaluation** -- for each side, which specific arguments you "
        "accept and which you reject, and why.\n"
        "4. **Final Judgment** -- the court's holding and concrete result.\n\n"
        f"{GROUNDING_REMINDER}"
    )
    assistant = (
        "## 1. Arguments FOR (Prosecution)\n"
        f"{pc.arguments_for}\n\n"
        "## 2. Arguments AGAINST (Defense)\n"
        f"{pc.arguments_against}\n\n"
        "## 3. Evaluation\n\n"
        "### Reasoning to Accept Arguments\n"
        f"{pc.acceptance_logic}\n\n"
        "### Reasoning to Reject Arguments\n"
        f"{pc.rejection_logic}\n\n"
        "## 4. Final Judgment\n"
        f"{pc.final_conclusion}"
    )
    messages = [_msg("system", SYSTEM_PROMPT), _msg("user", user), _msg("assistant", assistant)]
    return _example(pc.case_id, "full_chain_single_turn", messages)


# --------------------------------------------------------------------------- #
# Task E -- Full chain, multi-turn (hierarchical dialogue: 3 assistant turns)
# --------------------------------------------------------------------------- #

def build_task_full_chain_multi_turn(pc: ParsedCase) -> dict:
    context = build_context_block(pc)

    turn1_user = (
        "Below are the unified facts, key actors, evidentiary timeline, "
        "claims, cited precedent, and statutory provisions relevant to a "
        "criminal appeal.\n\n"
        f"{context}\n\n"
        "Using only this material, construct the strongest arguments FOR the "
        "prosecution's case and the strongest arguments AGAINST it (for the "
        f"defense/appellant). {GROUNDING_REMINDER}"
    )
    turn1_assistant = build_arguments_block(pc)

    turn2_user = (
        "Now evaluate the arguments you just made, the way an appellate "
        "court would. For each side, decide which specific arguments are "
        "persuasive enough to accept and which must be rejected, and explain "
        "the legal reasoning behind each determination."
    )
    turn2_assistant = build_evaluation_block(pc)

    turn3_user = (
        "Based on your evaluation, deliver the court's final holding. State "
        "clearly whether the prosecution's case is upheld or fails, explain "
        "the decisive reasoning, and give the concrete result/order."
    )
    turn3_assistant = pc.final_conclusion

    messages = [
        _msg("system", SYSTEM_PROMPT),
        _msg("user", turn1_user),
        _msg("assistant", turn1_assistant),
        _msg("user", turn2_user),
        _msg("assistant", turn2_assistant),
        _msg("user", turn3_user),
        _msg("assistant", turn3_assistant),
    ]
    return _example(pc.case_id, "full_chain_multi_turn", messages)


# --------------------------------------------------------------------------- #
# Task F -- Verdict prediction (short-form, good for fast eval / calibration)
# --------------------------------------------------------------------------- #

def build_task_verdict_prediction(pc: ParsedCase) -> Optional[dict]:
    if not pc.result_short:
        return None
    context = build_context_block(pc)
    args = build_arguments_block(pc)
    user = (
        "Here is the case material along with the arguments for and against "
        "the prosecution's case.\n\n"
        f"{context}\n\n"
        f"{args}\n\n"
        "In 1-3 sentences, predict the court's bottom-line result (e.g., "
        "conviction affirmed, conviction set aside and appellant acquitted, "
        "remanded for retrial) and the single decisive reason for it. Do not "
        "restate the arguments -- give only the predicted result and its "
        "core reason."
    )
    assistant = pc.result_short
    messages = [_msg("system", SYSTEM_PROMPT), _msg("user", user), _msg("assistant", assistant)]
    return _example(pc.case_id, "verdict_prediction", messages)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

TASK_BUILDERS = {
    "arguments_generation": build_task_arguments_generation,
    "argument_evaluation": build_task_argument_evaluation,
    "final_judgment": build_task_final_judgment,
    "full_chain_single_turn": build_task_full_chain_single_turn,
    "full_chain_multi_turn": build_task_full_chain_multi_turn,
    "verdict_prediction": build_task_verdict_prediction,
}


def build_examples_for_case(pc: ParsedCase, task_names: list[str]) -> list[dict]:
    """Return all requested SFT examples for one case. Skips a task silently
    if the case is missing the sections that task needs."""
    examples = []
    needed_sections = {
        "arguments_generation": ["facts", "law", "arguments_for", "arguments_against"],
        "argument_evaluation": ["facts", "law", "arguments_for", "arguments_against",
                                  "acceptance_logic", "rejection_logic"],
        "final_judgment": ["facts", "law", "arguments_for", "arguments_against",
                            "acceptance_logic", "rejection_logic", "final_conclusion"],
        "full_chain_single_turn": ["facts", "law", "arguments_for", "arguments_against",
                                    "acceptance_logic", "rejection_logic", "final_conclusion"],
        "full_chain_multi_turn": ["facts", "law", "arguments_for", "arguments_against",
                                    "acceptance_logic", "rejection_logic", "final_conclusion"],
        "verdict_prediction": ["facts", "law", "arguments_for", "arguments_against"],
    }
    for name in task_names:
        builder = TASK_BUILDERS[name]
        reqs = needed_sections[name]
        if any(not getattr(pc, r, "").strip() if r != "final_conclusion" else not pc.final_conclusion.strip()
               for r in reqs):
            continue
        ex = builder(pc)
        if ex is not None:
            examples.append(ex)
    return examples
