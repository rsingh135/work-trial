"""Per-log schema registry.

The raw XES is the source of truth; this file records *how we read it*. Every choice here
is justified in PROGRESS.md (search for the log name). Nothing in the registry is inferred
automatically at training time — a reviewer can read this file and know exactly which raw
fields feed the models.

Field roles
-----------
activity_keys   : event attributes fused (joined with "|") to form the activity label.
event_cat       : low/medium-cardinality event attributes used as categorical inputs.
event_highcard  : high-cardinality event attributes; kept but frequency-thresholded to UNK.
case_cat        : categorical case (trace) attributes known at case start.
case_num        : numeric case attributes known at case start (log1p-scaled downstream).
excluded        : fields we deliberately do not use, with reason.
terminal        : activity labels that denote a completed case (used for censoring logic).
censored        : whether the whole log should be treated as right-censored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BUNDLE = Path("researcher_work_trial_bundle")  # the reference bundle is the single source of truth
LOG_DIR = BUNDLE / "data"


@dataclass(frozen=True)
class LogSchema:
    name: str
    file: str
    family: str  # "bpi2013" | "bpi2020"
    activity_keys: tuple[str, ...]
    event_cat: tuple[str, ...] = ()
    event_highcard: tuple[str, ...] = ()
    case_cat: tuple[str, ...] = ()
    case_num: tuple[str, ...] = ()
    excluded: dict[str, str] = field(default_factory=dict)
    terminal: tuple[str, ...] = ()
    censored: bool = False
    local_tz: str = "Europe/Amsterdam"  # both BPI2013 (Volvo IT, SE) and BPI2020 (TU/e, NL) are CET/CEST
    notes: str = ""

    @property
    def path(self) -> Path:
        return LOG_DIR / self.file


_BPI2013_EXCLUDED = {
    "org:resource": "person names; high-cardinality, privacy-adjacent, and resource-level "
    "routing is not stable over time (used only as high-card input with UNK threshold)",
    "resource country": "near-duplicate of org:group/organization line; keep org:group instead",
    "organization country": "same reason; also key is misspelled in open_problems "
    "('oranization country') — treated as the same field via alias",
    "organization involved": "redundant with org:group (function line) in Incidents",
}

BPI2013_INCIDENTS = LogSchema(
    name="bpi2013_incidents",
    file="bpi2013/BPI_Challenge_2013_incidents.xes.gz",
    family="bpi2013",
    activity_keys=("concept:name", "lifecycle:transition"),
    event_cat=("org:group", "org:role", "impact", "organization involved"),
    event_highcard=("org:resource", "product"),
    excluded={k: v for k, v in _BPI2013_EXCLUDED.items() if k not in ("org:resource",)},
    terminal=("Completed|Closed", "Completed|Cancelled"),
    notes="Volvo IT incident management (VINST). Activity = status|substatus. Snapshot log: "
    "96% of cases start Apr-May 2012 and 98% of Closed events are in May 2012 (batch "
    "auto-close). 25% of cases end in Completed|In Call without a Closed event: treated as "
    "incomplete (right-censored within the log) for suffix/remaining-time targets.",
)

BPI2013_CLOSED = LogSchema(
    name="bpi2013_closed_problems",
    file="bpi2013/BPI_Challenge_2013_closed_problems.xes.gz",
    family="bpi2013",
    activity_keys=("concept:name", "lifecycle:transition"),
    event_cat=("org:group", "org:role", "impact", "organization involved"),
    event_highcard=("org:resource", "product"),
    excluded={k: v for k, v in _BPI2013_EXCLUDED.items() if k not in ("org:resource",)},
    terminal=("Completed|Closed", "Completed|Cancelled"),
    notes="Problem management, closed problems only. Audit: 100% of cases end in Completed|Closed.",
)

BPI2013_OPEN = LogSchema(
    name="bpi2013_open_problems",
    file="bpi2013/BPI_Challenge_2013_open_problems.xes.gz",
    family="bpi2013",
    activity_keys=("concept:name", "lifecycle:transition"),
    event_cat=("org:group", "org:role", "impact"),
    event_highcard=("org:resource", "product"),
    excluded={
        "oranization country": "misspelled key (sic) in this log only; excluded like its sibling",
        **{k: v for k, v in _BPI2013_EXCLUDED.items() if k not in ("org:resource",)},
    },
    terminal=("Completed|Closed",),
    censored=True,
    notes="Open problems at extraction time: right-censored. Remaining-time/suffix/outcome "
    "targets are masked; next-event targets on observed prefixes remain valid.",
)

# --- BPI 2020 -------------------------------------------------------------------------
# All five logs share event attrs: id, org:resource (anonymised to STAFF MEMBER / SYSTEM),
# concept:name, time:timestamp, org:role. Case attributes differ per log.

_BPI2020_EV_EXCL = {
    "id": "event identifier; encodes sub-process origin (st_step/dd_/rv_) and ordering → leakage",
}

BPI2020_DOMESTIC = LogSchema(
    name="bpi2020_domestic",
    file="bpi2020/DomesticDeclarations.xes.gz",
    family="bpi2020",
    activity_keys=("concept:name",),
    event_cat=("org:role",),
    event_highcard=("org:resource",),
    case_num=("Amount",),
    excluded={
        **_BPI2020_EV_EXCL,
        "BudgetNumber": "identifier-like, high-card",
        "DeclarationNumber": "identifier",
    },
    terminal=("Payment Handled", "Declaration REJECTED by EMPLOYEE", "Declaration REJECTED by MISSING"),
    notes="Simplest BPI2020 log: submit → approvals → payment. Audit: 98% end in Payment "
    "Handled; 134 cases end in SAVED by EMPLOYEE (never submitted → incomplete).",
)

BPI2020_INTERNATIONAL = LogSchema(
    name="bpi2020_international",
    file="bpi2020/InternationalDeclarations.xes.gz",
    family="bpi2020",
    activity_keys=("concept:name",),
    event_cat=("org:role",),
    event_highcard=("org:resource",),
    case_cat=("Permit OrganizationalEntity",),
    case_num=("Amount", "RequestedAmount", "Permit RequestedBudget"),
    excluded={
        **_BPI2020_EV_EXCL,
        "OriginalAmount": "equals Amount before adjustment; AdjustedAmount reveals outcome",
        "AdjustedAmount": "outcome-revealing (post-hoc adjustment) → leakage",
        "Permit ID/Permit id/travel permit number/Permit travel permit number": "identifiers",
        "DeclarationNumber/BudgetNumber/Permit BudgetNumber/Permit TaskNumber/"
        "Permit ProjectNumber/Permit ActivityNumber": "identifiers / mostly UNKNOWN",
    },
    terminal=("Payment Handled", "Declaration REJECTED by EMPLOYEE", "Declaration REJECTED by MISSING",
              "Permit REJECTED by EMPLOYEE", "Permit REJECTED by MISSING"),
    notes="Longer traces; permit + trip events precede declaration; more variants. Audit: 593 "
    "cases end in 'End trip' (trip done, no declaration yet) → incomplete.",
)

BPI2020_PERMIT = LogSchema(
    name="bpi2020_permit",
    file="bpi2020/PermitLog.xes.gz",
    family="bpi2020",
    activity_keys=("concept:name",),
    event_cat=("org:role",),
    event_highcard=("org:resource",),
    case_cat=("OrganizationalEntity",),
    case_num=("RequestedBudget",),
    excluded={
        **_BPI2020_EV_EXCL,
        "TotalDeclared/Overspent/OverspentAmount": "computed at case end → outcome leakage",
        "dec_id_*/DeclarationNumber_*/RequestedAmount_*": "per-declaration expansions (170+ "
        "sparse columns), known only after declarations exist → leakage + sparsity",
        "ProjectNumber/TaskNumber/ActivityNumber/BudgetNumber/travel permit number/id": "identifiers",
    },
    terminal=("Payment Handled", "Permit REJECTED by EMPLOYEE", "Permit REJECTED by MISSING",
              "Declaration REJECTED by EMPLOYEE", "Request For Payment REJECTED by EMPLOYEE"),
    notes="Permit-centric log; a permit can spawn many declarations. Widest schema (176 raw "
    "columns). Audit: dec_id_*/DeclarationNumber_* have NMI=1.0 with terminal activity → "
    "confirmed leakage, excluded. 991 cases end in Send Reminder, 453 in End trip → incomplete.",
)

BPI2020_PREPAID = LogSchema(
    name="bpi2020_prepaid",
    file="bpi2020/PrepaidTravelCost.xes.gz",
    family="bpi2020",
    activity_keys=("concept:name",),
    event_cat=("org:role",),
    event_highcard=("org:resource",),
    case_cat=("OrganizationalEntity", "Cost Type"),
    case_num=("RequestedAmount", "Permit RequestedBudget"),
    excluded={
        **_BPI2020_EV_EXCL,
        "Rfp_id/RfpNumber/Task/Activity/Project/Permit *Number/Permit id": "identifiers",
    },
    terminal=("Payment Handled", "Request For Payment REJECTED by EMPLOYEE",
              "Request For Payment REJECTED by MISSING", "Permit REJECTED by MISSING"),
    notes="Prepaid travel costs (request-for-payment linked to permit).",
)

BPI2020_RFP = LogSchema(
    name="bpi2020_rfp",
    file="bpi2020/RequestForPayment.xes.gz",
    family="bpi2020",
    activity_keys=("concept:name",),
    event_cat=("org:role",),
    event_highcard=("org:resource",),
    case_cat=("OrganizationalEntity", "Cost Type"),
    case_num=("RequestedAmount",),
    excluded={
        **_BPI2020_EV_EXCL,
        "Rfp_id/RfpNumber/Task/Activity/Project": "identifiers",
    },
    terminal=("Payment Handled", "Request For Payment REJECTED by EMPLOYEE",
              "Request For Payment REJECTED by MISSING"),
    notes="Requests for payment not tied to a trip. Audit: case_Project has NMI 0.71 with "
    "terminal activity but 79 values over 6886 cases (small-sample inflation); excluded as "
    "identifier-like anyway.",
)

LOGS: dict[str, LogSchema] = {
    s.name: s
    for s in (
        BPI2013_INCIDENTS,
        BPI2013_CLOSED,
        BPI2013_OPEN,
        BPI2020_DOMESTIC,
        BPI2020_INTERNATIONAL,
        BPI2020_PERMIT,
        BPI2020_PREPAID,
        BPI2020_RFP,
    )
}


def get_schema(name: str) -> LogSchema:
    if name not in LOGS:
        raise KeyError(f"unknown log {name!r}; known: {sorted(LOGS)}")
    return LOGS[name]
