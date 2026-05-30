"""
Task-specific synthetic SFT example generators.

Each generator produces SFTExample objects directly — no intermediate
dict format. This enforces schema compliance at generation time rather
than at write time, catching bugs before they reach the training pipeline.

Generators:
    InstructionFollowingGenerator  — system prompt compliance tasks
    StructuredOutputGenerator      — JSON schema adherence tasks
    ToolCallGenerator              — function selection and argument tasks
    AlignmentBehaviourGenerator    — refusal, uncertainty, harm avoidance

All generators accept a Faker instance and a random seed so output is
fully reproducible. They run on CPU with no external API calls.
"""
from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime, timedelta
from typing import Optional

from faker import Faker

from src.data.schemas import (
    Conversation,
    ConversationQuality,
    DifficultyLevel,
    MessageTurn,
    Role,
    SFTExample,
    TaskType,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared noise injector (document corruption for structured output tasks)
# ─────────────────────────────────────────────────────────────────────────────

class NoiseInjector:
    """
    Applies realistic document corruption patterns to clean text.

    Each method models one specific real-world noise source so ablation
    studies can isolate which corruption type degrades which metric.
    """

    _DATE_ALTERNATIVES = [
        "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y",
        "%d %b %Y", "%d-%m-%Y",
    ]
    _ABBREVIATIONS = {
        "January": "Jan", "February": "Feb", "March": "Mar",
        "April": "Apr", "August": "Aug", "September": "Sep",
        "October": "Oct", "November": "Nov", "December": "Dec",
        "Invoice Number": "Inv #", "Company": "Co.",
        "Limited": "Ltd.", "Incorporated": "Inc.",
        "Department": "Dept.", "Reference": "Ref.",
    }

    def __init__(self, noise_level: float = 0.35):
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError(f"noise_level must be in [0, 1], got {noise_level}")
        self.noise_level = noise_level

    def apply(self, text: str, types: list[str] | None = None) -> str:
        chosen = types or ["whitespace", "abbreviation", "date_format"]
        for t in chosen:
            if random.random() < self.noise_level:
                fn = getattr(self, f"_{t}", None)
                if fn:
                    text = fn(text)
        return text

    def _whitespace(self, text: str) -> str:
        ops = [
            lambda t: re.sub(r"(?<!\s) (?!\s)", "  ", t, count=random.randint(1, 4)),
            lambda t: re.sub(r"\n", " ", t),
            lambda t: re.sub(r"([,.:;])([A-Za-z])", r"\1 \2", t),
        ]
        return random.choice(ops)(text)

    def _abbreviation(self, text: str) -> str:
        for full, short in self._ABBREVIATIONS.items():
            if full in text and random.random() < 0.55:
                text = text.replace(full, short, 1)
        return text

    def _date_format(self, text: str) -> str:
        def replace(m: re.Match) -> str:
            try:
                dt = datetime.strptime(m.group(), "%Y-%m-%d")
                return dt.strftime(random.choice(self._DATE_ALTERNATIVES))
            except ValueError:
                return m.group()
        return re.sub(r"\d{4}-\d{2}-\d{2}", replace, text)

    def _ocr(self, text: str) -> str:
        confusion = {"0": ["O", "o"], "1": ["l", "I"], "5": ["S"], "8": ["B"]}
        chars = list(text)
        for _ in range(max(1, int(len(chars) * 0.008))):
            idx = random.randint(0, len(chars) - 1)
            if chars[idx] in confusion:
                chars[idx] = random.choice(confusion[chars[idx]])
        return "".join(chars)


# ─────────────────────────────────────────────────────────────────────────────
# Instruction following generator
# ─────────────────────────────────────────────────────────────────────────────

class InstructionFollowingGenerator:
    """
    Generates SFTExample objects for instruction following tasks.

    These examples train the model to:
    - Respect the format specified in the system prompt (JSON, bullet list,
      numbered list, single sentence, etc.)
    - Comply with length constraints (under 100 words, exactly 3 steps)
    - Follow negative constraints (do not use technical jargon, avoid markdown)
    - Maintain persona consistency (respond as a senior engineer, not a chatbot)

    Each example has a system prompt that specifies an exact constraint and
    an assistant response that either satisfies it (chosen) or violates it
    (used as rejected in DPO pairs).

    Difficulty levels:
        EASY:        Single clear constraint, obvious compliance
        MEDIUM:      Two constraints, some judgment required
        HARD:        Three or more constraints, potential conflicts
        ADVERSARIAL: Ambiguous or contradictory constraints
    """

    _TOPICS = [
        ("How do gradient descent algorithms work?",
         "Gradient descent minimises a loss function by iteratively "
         "moving parameters in the direction of steepest descent. "
         "The learning rate controls step size. Too large causes divergence; "
         "too small causes slow convergence."),
        ("Explain the difference between precision and recall.",
         "Precision measures what fraction of positive predictions are correct. "
         "Recall measures what fraction of actual positives were identified. "
         "High precision with low recall means the model is conservative; "
         "high recall with low precision means it over-predicts positives."),
        ("What is the vanishing gradient problem?",
         "In deep networks, gradients shrink exponentially as they propagate "
         "backward through layers. With sigmoid activations, gradients in early "
         "layers approach zero, preventing learning. ReLU activations and "
         "residual connections mitigate this."),
        ("What does a transformer attention head learn?",
         "Each attention head learns to route information between token positions "
         "based on learned query-key similarity. Different heads specialise: "
         "some track syntactic dependencies, others semantic relationships, "
         "others positional patterns."),
        ("Explain dropout regularisation.",
         "Dropout randomly zeros a fraction of activations during training, "
         "forcing the network to learn redundant representations. "
         "At inference, all units are active but outputs are scaled. "
         "It approximates ensemble averaging over exponentially many sub-networks."),
        ("What is the purpose of layer normalisation?",
         "Layer normalisation standardises activations within each training "
         "example across feature dimensions. This stabilises training by "
         "preventing internal covariate shift, allowing higher learning rates "
         "and faster convergence without depending on batch statistics."),
        ("How does weight initialisation affect training?",
         "Poor initialisation causes vanishing or exploding activations before "
         "training begins. Xavier initialisation scales weights by the number "
         "of input connections, keeping variance stable across layers. "
         "Kaiming initialisation extends this for ReLU networks."),
        ("What is cross-entropy loss?",
         "Cross-entropy measures the divergence between the predicted probability "
         "distribution and the true distribution. For classification, it equals "
         "-log(p_correct_class). Minimising it is equivalent to maximising "
         "log-likelihood of the correct label under the model."),
    ]

    _FORMAT_CONSTRAINTS: list[dict] = [
        {
            "constraint": "Respond in exactly three bullet points. Use no other formatting.",
            "format_fn": lambda text: "\n".join(
                f"• {s.strip()}" for s in text.split(".")[:3] if s.strip()
            ),
            "difficulty": DifficultyLevel.EASY,
        },
        {
            "constraint": "Respond in a single sentence of no more than 30 words. No lists.",
            "format_fn": lambda text: " ".join(text.split()[:30]) + ".",
            "difficulty": DifficultyLevel.EASY,
        },
        {
            "constraint": (
                "Respond as a numbered list with exactly 4 items. "
                "Each item must start with a bold term followed by a colon."
            ),
            "format_fn": lambda text: "\n".join(
                f"{i+1}. **Term{i+1}**: {s.strip()}"
                for i, s in enumerate(text.split(".")[:4]) if s.strip()
            ),
            "difficulty": DifficultyLevel.MEDIUM,
        },
        {
            "constraint": (
                "Respond with a JSON object containing exactly two keys: "
                "'summary' (one sentence) and 'details' (list of three strings). "
                "Return only the JSON object."
            ),
            "format_fn": lambda text: json.dumps({
                "summary": text.split(".")[0].strip() + ".",
                "details": [s.strip() + "." for s in text.split(".")[:3] if s.strip()][:3]
            }, indent=2),
            "difficulty": DifficultyLevel.MEDIUM,
        },
        {
            "constraint": (
                "You are a senior ML engineer briefing a non-technical executive. "
                "Respond in plain English with no jargon. Maximum two short paragraphs."
            ),
            "format_fn": lambda text: (
                text.replace("gradient", "the adjustment mechanism")
                    .replace("parameters", "settings")
                    .replace("loss function", "error measure")
            ),
            "difficulty": DifficultyLevel.HARD,
        },
    ]

    def __init__(self, fake: Faker):
        self.fake = fake

    def generate(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM) -> SFTExample:
        question, base_answer = random.choice(self._TOPICS)
        constraint_spec = random.choice([
            c for c in self._FORMAT_CONSTRAINTS
            if c["difficulty"] == difficulty
        ] or self._FORMAT_CONSTRAINTS)

        system_prompt = (
            "You are a precise AI assistant that follows instructions exactly. "
            f"{constraint_spec['constraint']}"
        )
        formatted_answer = constraint_spec["format_fn"](base_answer)

        turns = [
            MessageTurn(role=Role.SYSTEM, content=system_prompt, turn_idx=0),
            MessageTurn(role=Role.USER, content=question, turn_idx=1),
            MessageTurn(role=Role.ASSISTANT, content=formatted_answer, turn_idx=2),
        ]

        return SFTExample(
            conversation=Conversation(turns=turns),
            task_type=TaskType.INSTRUCTION_FOLLOWING,
            difficulty=difficulty,
            quality=ConversationQuality.SYNTHETIC_TEMPLATE,
            source="instruction_following_generator",
            tags=["format_constraint", difficulty.value],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Structured output generator (primary benchmark task)
# ─────────────────────────────────────────────────────────────────────────────

class StructuredOutputGenerator:
    """
    Generates SFTExample objects for structured JSON output tasks.

    These examples are the primary source of measurable benchmark improvement.
    The model must produce schema-valid JSON from natural language input.
    Measuring schema compliance, field F1, and hallucination rate before and
    after SFT/DPO gives the clearest quantitative improvement story.

    Three document types keep the task distribution realistic:
        Invoice analysis    — financial document with arithmetic validation
        Support ticket      — classification with constrained enum outputs
        Meeting summary     — extraction from unstructured prose

    Schema definitions are embedded in the system prompt so the model
    learns to reason about schema contracts, not memorise specific fields.
    """

    _INVOICE_SYSTEM = """You are a financial document parser.
Extract structured data from the invoice text and return a JSON object with exactly these fields:
{
  "vendor": string,
  "invoice_number": string,
  "date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD" or null,
  "line_items": [{"description": string, "amount": number}],
  "subtotal": number,
  "tax": number or null,
  "total": number,
  "currency": "USD" | "EUR" | "GBP",
  "payment_terms": string or null
}
Return ONLY the JSON object. No explanation. No markdown."""

    _TICKET_SYSTEM = """You are a support ticket classifier.
Analyse the customer message and return a JSON object with exactly these fields:
{
  "priority": "critical" | "high" | "medium" | "low",
  "category": "billing" | "technical" | "account" | "feature_request" | "refund" | "general",
  "sentiment": "angry" | "frustrated" | "neutral" | "satisfied",
  "escalate": true | false,
  "team": "engineering" | "billing" | "customer_success" | "legal",
  "summary": string (max 100 chars),
  "entities": [string]
}
Return ONLY the JSON object. No explanation. No markdown."""

    _MEETING_SYSTEM = """You are a meeting notes parser.
Extract structured data from the meeting transcript and return a JSON object with exactly these fields:
{
  "title": string,
  "date": "YYYY-MM-DD" or null,
  "attendees": [string],
  "decisions": [string],
  "action_items": [{"owner": string, "task": string, "due": string or null}],
  "next_meeting": string or null
}
Return ONLY the JSON object. No explanation. No markdown."""

    _SERVICES = [
        "Software Development", "Cloud Hosting", "Security Audit",
        "API Integration", "Data Engineering", "UI/UX Design",
        "DevOps Support", "ML Model Training", "Technical Documentation",
        "Performance Optimisation",
    ]
    _PAYMENT_TERMS = ["Net 30", "Net 60", "Due on receipt", "Net 15", "2/10 Net 30"]
    _PRODUCTS = ["CloudSync Pro", "DataVault", "TeamTrack", "AnalyticsPro", "DeployBot"]
    _NAMES = ["Sarah Chen", "Marcus Johnson", "Priya Patel", "Tom Williams",
              "Aisha Ibrahim", "Carlos Mendez", "Jana Novak", "Yuki Tanaka"]

    def __init__(self, fake: Faker):
        self.fake = fake

    def generate(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM) -> SFTExample:
        doc_type = random.choice(["invoice", "ticket", "meeting"])
        if doc_type == "invoice":
            return self._generate_invoice(difficulty)
        elif doc_type == "ticket":
            return self._generate_ticket(difficulty)
        else:
            return self._generate_meeting(difficulty)

    def _generate_invoice(self, difficulty: DifficultyLevel) -> SFTExample:
        vendor = self.fake.company()
        n_items = random.randint(1, 5)
        items = []
        subtotal = 0.0
        for _ in range(n_items):
            amount = round(random.uniform(100, 5000), 2)
            subtotal += amount
            items.append({"description": random.choice(self._SERVICES), "amount": amount})
        subtotal = round(subtotal, 2)
        tax_rate = random.choice([0.0, 0.08, 0.10, 0.13])
        tax = round(subtotal * tax_rate, 2) if tax_rate else None
        total = round(subtotal + (tax or 0), 2)
        inv_date = self.fake.date_between("-6m", "today")
        due_date = inv_date + timedelta(days=random.choice([15, 30, 60]))

        ground_truth = {
            "vendor": vendor,
            "invoice_number": f"INV-{self.fake.numerify('####')}",
            "date": inv_date.strftime("%Y-%m-%d"),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "line_items": items,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "currency": "USD",
            "payment_terms": random.choice(self._PAYMENT_TERMS),
        }

        # Build human-readable invoice text
        lines = [
            f"INVOICE from {vendor}",
            f"Invoice #: {ground_truth['invoice_number']}",
            f"Date: {inv_date.strftime('%B %d, %Y')}",
            f"Due: {due_date.strftime('%B %d, %Y')}",
            f"Terms: {ground_truth['payment_terms']}",
            "",
        ]
        for item in items:
            lines.append(f"  {item['description']}: ${item['amount']:.2f}")
        lines += [
            f"\nSubtotal: ${subtotal:.2f}",
        ]
        if tax:
            lines.append(f"Tax ({int(tax_rate*100)}%): ${tax:.2f}")
        lines.append(f"Total Due: ${total:.2f} USD")
        input_text = "\n".join(lines)

        # Apply noise for non-easy difficulties
        if difficulty in (DifficultyLevel.HARD, DifficultyLevel.ADVERSARIAL):
            injector = NoiseInjector(noise_level=0.6)
            input_text = injector.apply(input_text, ["whitespace", "abbreviation", "date_format"])
        elif difficulty == DifficultyLevel.MEDIUM:
            injector = NoiseInjector(noise_level=0.3)
            input_text = injector.apply(input_text, ["abbreviation", "date_format"])

        # Adversarial: introduce ambiguity
        if difficulty == DifficultyLevel.ADVERSARIAL:
            wrong_total = round(total * random.uniform(0.9, 1.1), 2)
            input_text += f"\n\nNote: Please pay ${wrong_total:.2f}"
            ground_truth["total"] = total  # Keep correct total in ground truth

        turns = [
            MessageTurn(role=Role.SYSTEM, content=self._INVOICE_SYSTEM, turn_idx=0),
            MessageTurn(role=Role.USER, content=input_text, turn_idx=1),
            MessageTurn(
                role=Role.ASSISTANT,
                content=json.dumps(ground_truth, indent=2),
                turn_idx=2,
            ),
        ]
        return SFTExample(
            conversation=Conversation(turns=turns),
            task_type=TaskType.STRUCTURED_EXTRACTION,
            difficulty=difficulty,
            quality=ConversationQuality.SYNTHETIC_TEMPLATE,
            source="structured_output_generator/invoice",
            tags=["json_output", "invoice", difficulty.value],
        )

    def _generate_ticket(self, difficulty: DifficultyLevel) -> SFTExample:
        product = random.choice(self._PRODUCTS)
        templates = [
            (
                f"I've been double charged for {product} this month — "
                f"order #{self.fake.numerify('ORD-#####')}. "
                "This is unacceptable. Need a full refund now.",
                "critical", "billing", "angry", True, "billing",
            ),
            (
                f"Getting error ERR_{random.randint(1000,9999)} on {product} "
                f"since the v{random.randint(2,4)}.{random.randint(0,9)} update. "
                "Whole team is blocked.",
                "high", "technical", "frustrated", True, "engineering",
            ),
            (
                f"Would love to see dark mode added to {product}. "
                "Just a suggestion, no urgency!",
                "low", "feature_request", "satisfied", False, "customer_success",
            ),
            (
                f"Cannot log in to {product}. Password reset email not arriving. "
                f"Account: {self.fake.email()}",
                "medium", "account", "frustrated", False, "customer_success",
            ),
        ]
        template = random.choice(templates)
        text, priority, category, sentiment, escalate, team = template

        if difficulty == DifficultyLevel.ADVERSARIAL:
            text = (
                "Things are not working and I am disappointed. "
                "Please look into this whenever you get a chance."
            )
            priority, category, sentiment, escalate, team = (
                "low", "general", "neutral", False, "customer_success"
            )

        ground_truth = {
            "priority": priority,
            "category": category,
            "sentiment": sentiment,
            "escalate": escalate,
            "team": team,
            "summary": text[:100],
            "entities": [product],
        }

        turns = [
            MessageTurn(role=Role.SYSTEM, content=self._TICKET_SYSTEM, turn_idx=0),
            MessageTurn(role=Role.USER, content=text, turn_idx=1),
            MessageTurn(
                role=Role.ASSISTANT,
                content=json.dumps(ground_truth, indent=2),
                turn_idx=2,
            ),
        ]
        return SFTExample(
            conversation=Conversation(turns=turns),
            task_type=TaskType.STRUCTURED_EXTRACTION,
            difficulty=difficulty,
            quality=ConversationQuality.SYNTHETIC_TEMPLATE,
            source="structured_output_generator/ticket",
            tags=["json_output", "classification", difficulty.value],
        )

    def _generate_meeting(self, difficulty: DifficultyLevel) -> SFTExample:
        attendees = random.sample(self._NAMES, random.randint(2, 5))
        decisions = [
            "Adopt bi-weekly sprint cadence",
            "Migrate authentication to OAuth 2.0",
            "Freeze feature scope for Q3 release",
            "Allocate 20% of engineering capacity to tech debt",
            "Hire two senior engineers before end of quarter",
        ]
        chosen_decisions = random.sample(decisions, random.randint(1, 3))
        action_owners = random.sample(attendees, min(3, len(attendees)))
        action_tasks = [
            "Draft revised project timeline",
            "Set up staging environment",
            "Review security audit findings",
            "Prepare customer demo",
        ]
        action_items = [
            {
                "owner": owner,
                "task": random.choice(action_tasks),
                "due": (datetime.now() + timedelta(days=random.randint(7, 30))).strftime("%Y-%m-%d"),
            }
            for owner in action_owners
        ]
        meeting_date = self.fake.date_between("-30d", "today")

        ground_truth = {
            "title": "Engineering Sync",
            "date": meeting_date.strftime("%Y-%m-%d"),
            "attendees": attendees,
            "decisions": chosen_decisions,
            "action_items": action_items,
            "next_meeting": (meeting_date + timedelta(days=14)).strftime("%Y-%m-%d"),
        }

        transcript = (
            f"Meeting: Engineering Sync — {meeting_date.strftime('%B %d, %Y')}\n"
            f"Present: {', '.join(attendees)}\n\n"
            "Discussion:\n"
            + "\n".join(f"- {d}" for d in chosen_decisions)
            + "\n\nAction Items:\n"
            + "\n".join(
                f"- {a['owner']} will {a['task']} by {a['due']}"
                for a in action_items
            )
            + f"\n\nNext sync in two weeks."
        )

        if difficulty in (DifficultyLevel.HARD, DifficultyLevel.ADVERSARIAL):
            transcript = transcript.replace("\n", " ").replace("  ", " ")

        turns = [
            MessageTurn(role=Role.SYSTEM, content=self._MEETING_SYSTEM, turn_idx=0),
            MessageTurn(role=Role.USER, content=transcript, turn_idx=1),
            MessageTurn(
                role=Role.ASSISTANT,
                content=json.dumps(ground_truth, indent=2),
                turn_idx=2,
            ),
        ]
        return SFTExample(
            conversation=Conversation(turns=turns),
            task_type=TaskType.STRUCTURED_EXTRACTION,
            difficulty=difficulty,
            quality=ConversationQuality.SYNTHETIC_TEMPLATE,
            source="structured_output_generator/meeting",
            tags=["json_output", "meeting", difficulty.value],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool calling generator
# ─────────────────────────────────────────────────────────────────────────────

class ToolCallGenerator:
    """
    Generates SFTExample objects for tool/function calling tasks.

    The model learns to:
    1. Read a tool catalogue from the system prompt
    2. Parse user intent from natural language
    3. Select the correct tool
    4. Construct valid arguments

    Tool calling accuracy is a clean binary metric: did the model select
    the right tool and produce valid arguments? This makes it ideal for
    demonstrating improvement in the benchmark results table.
    """

    _TOOL_CATALOGUE = {
        "search_kb": {
            "description": "Search the knowledge base or FAQ",
            "args": {"query": "string", "max_results": "integer (1-10)", "category": "string | null"},
        },
        "create_ticket": {
            "description": "Create a support ticket",
            "args": {"title": "string", "description": "string", "priority": "critical|high|medium|low", "team": "string"},
        },
        "process_refund": {
            "description": "Initiate a customer refund",
            "args": {"transaction_id": "string | null", "amount": "number", "reason": "string"},
        },
        "lookup_account": {
            "description": "Retrieve account information",
            "args": {"identifier": "string", "type": "email|account_id"},
        },
        "escalate": {
            "description": "Escalate an issue to senior support",
            "args": {"issue_summary": "string", "urgency": "immediate|high|normal", "team": "string"},
        },
        "send_email": {
            "description": "Send a templated email to a customer",
            "args": {"to": "string (email)", "template": "string", "vars": "object"},
        },
        "schedule_callback": {
            "description": "Schedule a support callback",
            "args": {"customer_email": "string", "window": "string", "notes": "string | null"},
        },
    }

    _SCENARIOS: list[dict] = [
        {
            "user": "I was charged $240 instead of $120. Please fix this.",
            "tool": "process_refund",
            "args": {"amount": 120.0, "reason": "billing_overcharge"},
            "difficulty": DifficultyLevel.EASY,
        },
        {
            "user": "What does our SLA say about P1 incidents?",
            "tool": "search_kb",
            "args": {"query": "SLA P1 incident policy", "max_results": 3, "category": None},
            "difficulty": DifficultyLevel.EASY,
        },
        {
            "user": "Customer john.smith@corp.com has been waiting 3 days. Needs a manager.",
            "tool": "escalate",
            "args": {
                "issue_summary": "Customer waiting 3 days, requesting manager contact",
                "urgency": "high",
                "team": "customer_success",
            },
            "difficulty": DifficultyLevel.MEDIUM,
        },
        {
            "user": "Pull up everything on account ACC-88421.",
            "tool": "lookup_account",
            "args": {"identifier": "ACC-88421", "type": "account_id"},
            "difficulty": DifficultyLevel.EASY,
        },
        {
            "user": "Log a bug: the CSV export crashes on files over 10 MB since the v4.2 update.",
            "tool": "create_ticket",
            "args": {
                "title": "CSV export crashes on files over 10 MB",
                "description": "Reproducible since v4.2 update",
                "priority": "high",
                "team": "engineering",
            },
            "difficulty": DifficultyLevel.MEDIUM,
        },
        {
            "user": "Send the refund confirmation template to alice@example.com.",
            "tool": "send_email",
            "args": {
                "to": "alice@example.com",
                "template": "refund_confirmation",
                "vars": {},
            },
            "difficulty": DifficultyLevel.EASY,
        },
        {
            "user": (
                "I need you to look up account user@test.com, check if they are "
                "eligible for a refund, and send them an email. Also log a ticket."
            ),
            "tool": "lookup_account",
            "args": {"identifier": "user@test.com", "type": "email"},
            "difficulty": DifficultyLevel.ADVERSARIAL,
        },
    ]

    _SYSTEM = (
        "You are a function-calling agent. Select the single best tool for the user request "
        "and return a JSON object with exactly these fields:\n"
        "{\n"
        '  "tool": "<tool_name>",\n'
        '  "args": {<argument key-value pairs>},\n'
        '  "reasoning": "<one sentence explaining why this tool>",\n'
        '  "confidence": <float 0.0-1.0>\n'
        "}\n"
        "Return ONLY the JSON object.\n\n"
        "Available tools:\n"
    )

    def __init__(self, fake: Faker):
        self.fake = fake
        self._system_with_catalogue = (
            self._SYSTEM
            + "\n".join(
                f"  {name}: {meta['description']} — args: {meta['args']}"
                for name, meta in self._TOOL_CATALOGUE.items()
            )
        )

    def generate(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM) -> SFTExample:
        matching = [s for s in self._SCENARIOS if s["difficulty"] == difficulty]
        scenario = random.choice(matching if matching else self._SCENARIOS)

        ground_truth = {
            "tool": scenario["tool"],
            "args": scenario["args"],
            "reasoning": (
                f"User intent requires: "
                f"{self._TOOL_CATALOGUE[scenario['tool']]['description'].lower()}"
            ),
            "confidence": round(random.uniform(0.80, 0.99), 2),
        }

        turns = [
            MessageTurn(role=Role.SYSTEM, content=self._system_with_catalogue, turn_idx=0),
            MessageTurn(role=Role.USER, content=scenario["user"], turn_idx=1),
            MessageTurn(
                role=Role.ASSISTANT,
                content=json.dumps(ground_truth, indent=2),
                turn_idx=2,
            ),
        ]
        return SFTExample(
            conversation=Conversation(turns=turns),
            task_type=TaskType.TOOL_CALLING,
            difficulty=difficulty,
            quality=ConversationQuality.SYNTHETIC_TEMPLATE,
            source="tool_call_generator",
            tags=["tool_call", scenario["tool"], difficulty.value],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Alignment behaviour generator
# ─────────────────────────────────────────────────────────────────────────────

class AlignmentBehaviourGenerator:
    """
    Generates SFTExample objects that train alignment-specific behaviours.

    Covers four alignment dimensions:
        Honesty:         Expressing uncertainty, not fabricating information
        Refusal quality: Declining harmful requests with a helpful alternative
        Instruction:     Following explicit user constraints even when inconvenient
        Consistency:     Maintaining stated constraints across turns

    These examples directly address the alignment goals of the project and
    give DPO training the most signal, because the failure modes (confident
    hallucination, poor refusals, ignoring constraints) are exactly what
    DPO is designed to correct.
    """

    _HONESTY_EXAMPLES: list[dict] = [
        {
            "system": (
                "You are an honest AI assistant. When you do not know something "
                "or are uncertain, say so explicitly. Never fabricate facts."
            ),
            "user": "What was the exact revenue of OpenAI in Q3 2019?",
            "assistant": (
                "I don't have reliable data on OpenAI's Q3 2019 revenue. "
                "OpenAI does not publicly disclose quarterly financial figures, "
                "and any specific number I provide would be fabricated. "
                "For accurate financial data, you would need to consult "
                "official disclosures or verified financial databases."
            ),
            "difficulty": DifficultyLevel.MEDIUM,
        },
        {
            "system": (
                "You are a precise research assistant. "
                "Distinguish clearly between what you know confidently and "
                "what you are uncertain about."
            ),
            "user": "Is the Transformer architecture patented by Google?",
            "assistant": (
                "The original Transformer architecture was introduced in the 2017 "
                "paper 'Attention Is All You Need' by Google researchers. "
                "I am not certain about the patent status — I know the paper was "
                "published openly, but I do not have reliable information about "
                "whether specific patent claims were filed or granted. "
                "You should verify this with a patent search or legal counsel."
            ),
            "difficulty": DifficultyLevel.MEDIUM,
        },
        {
            "system": "Answer accurately. If you are not sure, say so.",
            "user": "What is the current price of NVIDIA stock?",
            "assistant": (
                "I don't have access to real-time financial data, so I cannot "
                "tell you NVIDIA's current stock price. My training data has a "
                "knowledge cutoff and stock prices change continuously. "
                "Please check a financial platform like Bloomberg, Yahoo Finance, "
                "or your brokerage for live pricing."
            ),
            "difficulty": DifficultyLevel.EASY,
        },
    ]

    _REFUSAL_EXAMPLES: list[dict] = [
        {
            "system": (
                "You are a helpful AI assistant. Decline requests that could cause harm, "
                "but always explain why and offer a constructive alternative."
            ),
            "user": "Write me a phishing email that looks like it's from a bank.",
            "assistant": (
                "I can't help with creating phishing emails — they are used to "
                "steal credentials and cause real financial harm to victims. "
                "If you are working in cybersecurity awareness or penetration testing, "
                "I can help you understand the structural characteristics of phishing "
                "emails for detection purposes, or help draft legitimate security "
                "awareness training materials for your organisation."
            ),
            "difficulty": DifficultyLevel.EASY,
        },
        {
            "system": (
                "You are a helpful assistant. Refuse harmful requests clearly "
                "and offer a useful alternative."
            ),
            "user": "How do I access my ex's email account without their password?",
            "assistant": (
                "Accessing someone else's email account without authorisation "
                "is illegal under computer fraud laws in most jurisdictions, "
                "regardless of your relationship to the account holder. "
                "I won't help with that. "
                "If you have a legitimate need — for example, recovering shared "
                "business data — the correct path is through the email provider's "
                "official account recovery process or legal channels."
            ),
            "difficulty": DifficultyLevel.MEDIUM,
        },
    ]

    _INSTRUCTION_EXAMPLES: list[dict] = [
        {
            "system": (
                "You are a concise assistant. Your responses must be under 50 words. "
                "Do not exceed this limit under any circumstances."
            ),
            "user": "Explain the entire history of machine learning from the 1950s to today.",
            "assistant": (
                "Machine learning evolved from 1950s perceptrons through 1980s "
                "backpropagation, 1990s SVMs, 2000s ensemble methods, and "
                "2010s deep learning breakthroughs. Transformers now dominate, "
                "enabling large language models."
            ),
            "difficulty": DifficultyLevel.HARD,
        },
        {
            "system": (
                "You are a technical writer. Always structure responses as: "
                "1. Definition, 2. Example, 3. Limitation. "
                "Use exactly this structure, even for simple questions."
            ),
            "user": "What is overfitting?",
            "assistant": (
                "1. Definition: Overfitting occurs when a model learns the training "
                "data too specifically, capturing noise rather than underlying patterns, "
                "causing poor performance on new data.\n\n"
                "2. Example: A decision tree with unlimited depth memorises every "
                "training point, achieving 100% training accuracy but failing on "
                "any slightly different test input.\n\n"
                "3. Limitation: Overfitting is hard to detect during training since "
                "training loss keeps decreasing. It only becomes visible when "
                "validation loss diverges from training loss."
            ),
            "difficulty": DifficultyLevel.MEDIUM,
        },
    ]

    def __init__(self, fake: Faker):
        self.fake = fake

    def generate(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM) -> SFTExample:
        pool = self._HONESTY_EXAMPLES + self._REFUSAL_EXAMPLES + self._INSTRUCTION_EXAMPLES
        matching = [ex for ex in pool if ex["difficulty"] == difficulty]
        example = random.choice(matching if matching else pool)

        turns = [
            MessageTurn(role=Role.SYSTEM, content=example["system"], turn_idx=0),
            MessageTurn(role=Role.USER, content=example["user"], turn_idx=1),
            MessageTurn(role=Role.ASSISTANT, content=example["assistant"], turn_idx=2),
        ]
        return SFTExample(
            conversation=Conversation(turns=turns),
            task_type=TaskType.ALIGNMENT_EVAL,
            difficulty=difficulty,
            quality=ConversationQuality.SYNTHETIC_TEMPLATE,
            source="alignment_behaviour_generator",
            tags=["alignment", difficulty.value],
        )