"""
Agent Class Structure — Python Skeleton
Maps to: Agent Component Blueprint (sections 1–19)

Notes:
- This is a framework-agnostic, testable skeleton. Methods are interfaces or light stubs.
- Prefer composition over inheritance; use ABCs/Protocols for pluggable strategy modules.
- Replace `pass` with your implementation and wire in your chosen providers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Protocol, Iterable, Callable, Union

# ------------------------------
# Section 4: Manifest & Contracts
# ------------------------------

class JsonSchemaRef(str):
    """Opaque reference to a JSON Schema (path/URL)."""

@dataclass
class ToolSpec:
    id: str
    description: str
    input_schema: JsonSchemaRef
    output_schema: JsonSchemaRef
    auth: str = "service"  # e.g., "oauth", "service", "user"
    retry_max: int = 2
    retry_backoff: str = "exp"
    safe: str = "read_only"  # e.g., "read_only" | "write" | "destructive"

@dataclass
class ModelRef:
    provider: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoutingConfig:
    quality_tier: str = "auto"
    budget_tokens_max: int = 120_000
    cost_per_task_max_usd: float = 0.20

@dataclass
class PolicyConfig:
    tool_use: str
    memory_read: bool = True
    memory_write: str = "selective"  # "always" | "never" | "selective"
    hitl_on: List[str] = field(default_factory=list)

@dataclass
class MemoryConfig:
    working_window_msgs: int = 20
    episodic_enabled: bool = True
    semantic_index: str = "faiss"
    semantic_k: int = 8
    decay_half_life_days: int = 14
    procedural_store: str = "procedures.db"

@dataclass
class SafetyConfig:
    content_policies: List[str]
    pii_blocklist: List[str]
    redact_outputs: bool = True

@dataclass
class TelemetryConfig:
    traces: bool = True
    metrics: List[str] = field(default_factory=lambda: ["latency", "token", "tool_success", "judge_score"])
    log_level: str = "info"
    pii_safe_logs: bool = True

@dataclass
class DeploymentConfig:
    envs: List[str] = field(default_factory=lambda: ["dev", "prod"])
    secrets: List[str] = field(default_factory=list)
    tenancy_mode: str = "per-org"

@dataclass
class AgentManifest:
    name: str
    persona_description: str
    audience: List[str]
    objectives_primary: List[str]
    constraints: List[str]
    policies: PolicyConfig
    routing: RoutingConfig
    input_schema: JsonSchemaRef
    output_schema: JsonSchemaRef
    redaction_rules: List[str]
    models_primary: ModelRef
    models_fallback: List[ModelRef]
    embeddings: Optional[ModelRef]
    memory: MemoryConfig
    tools: List[ToolSpec]
    safety: SafetyConfig
    telemetry: TelemetryConfig
    deployment: DeploymentConfig

# ------------------------------
# Section 7: Message & Event Protocols
# ------------------------------

@dataclass
class Citation:
    uri: str
    passage: Optional[str] = None
    hash: Optional[str] = None
    ts: Optional[str] = None

@dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]
    timeout_s: int = 30
    trace_id: Optional[str] = None

@dataclass
class ToolResult:
    id: str
    output: Any = None
    error: Optional[str] = None
    cost: Optional[Dict[str, Any]] = None

@dataclass
class Message:
    id: str
    ts: str
    role: str  # "user" | "assistant" | "system" | "tool"
    content: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    tool_calls: List[ToolCall] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    confidence: Optional[float] = None
    tenant: Optional[str] = None
    user: Optional[str] = None

@dataclass
class Event:
    name: str
    payload: Dict[str, Any]

# ------------------------------
# Section 2: Reasoning & Planning
# ------------------------------

@dataclass
class PlanStep:
    id: str
    action: str  # e.g., "retrieve", "call_tool:kpi.fetch", "generate", "verify"
    args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Plan:
    steps: List[PlanStep]
    stop_conditions: List[str] = field(default_factory=list)

class Planner(ABC):
    @abstractmethod
    def make_plan(self, msg: Message, ctx: "Context", router: "ModelRouter") -> Plan:
        pass

class Critic(ABC):
    @abstractmethod
    def critique(self, ctx: "Context") -> Tuple[bool, Optional[str]]:
        """Return (ok, reason)."""
        pass

# ------------------------------
# Section 3: State, Orchestration & Scheduling
# ------------------------------

@dataclass
class Context:
    conversation: List[Message] = field(default_factory=list)
    task: Dict[str, Any] = field(default_factory=dict)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    tool_state: Dict[str, Any] = field(default_factory=dict)
    error_state: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None

class Node(ABC):
    name: str
    @abstractmethod
    def run(self, ctx: Context) -> Context:
        pass

class Orchestrator(ABC):
    @abstractmethod
    def run(self, ctx: Context, plan: Plan) -> Context:
        pass

class FSMOrchestrator(Orchestrator):
    """A minimal finite-state style executor over plan steps."""
    def run(self, ctx: Context, plan: Plan) -> Context:
        for step in plan.steps:
            # Simple routing by step.action prefix
            if step.action.startswith("call_tool:"):
                tool_id = step.action.split(":", 1)[1]
                ctx = ToolExecutor.execute(tool_id, step.args, ctx)
            elif step.action == "retrieve":
                ctx = Retrieval.retrieve(step.args, ctx)
            elif step.action == "generate":
                ctx = Generation.generate(step.args, ctx)
            elif step.action == "verify":
                ok, reason = GlobalRegistry.critic.critique(ctx)
                if not ok:
                    ctx.error_state["verify"] = reason
                    break
        return ctx

# ------------------------------
# Section 5: Tools & Integrations
# ------------------------------

class Tool(ABC):
    spec: ToolSpec
    @abstractmethod
    def call(self, args: Dict[str, Any], ctx: Context) -> Any:
        pass

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.id] = tool

    def get(self, tool_id: str) -> Tool:
        if tool_id not in self._tools:
            raise KeyError(f"Tool not found: {tool_id}")
        return self._tools[tool_id]

class ToolExecutor:
    registry: ToolRegistry = ToolRegistry()

    @classmethod
    def execute(cls, tool_id: str, args: Dict[str, Any], ctx: Context) -> Context:
        tool = cls.registry.get(tool_id)
        try:
            output = tool.call(args, ctx)
            ctx.tool_state[tool_id] = {"last_output": output}
        except Exception as e:
            ctx.error_state[tool_id] = str(e)
        return ctx

# ------------------------------
# Section 6: Memory System
# ------------------------------

class WorkingMemory(ABC):
    @abstractmethod
    def context(self, msg: Message) -> Context:
        pass

class EpisodicMemory(ABC):
    @abstractmethod
    def record(self, event: Dict[str, Any]) -> None:
        pass

class SemanticMemory(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        pass

class ProceduralMemory(ABC):
    @abstractmethod
    def write_playbook(self, entry: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def recall(self, query: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
        pass

class MemorySystem:
    def __init__(self, working: WorkingMemory, episodic: EpisodicMemory,
                 semantic: SemanticMemory, procedural: ProceduralMemory):
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural

    # Convenience
    def context(self, msg: Message) -> Context:
        return self.working.context(msg)

# ------------------------------
# Retrieval & Generation Facades (Section 5/9)
# ------------------------------

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, ctx: Context, **kwargs) -> Dict[str, Any]:
        pass

class Reranker(ABC):
    @abstractmethod
    def rerank(self, items: List[Dict[str, Any]], ctx: Context) -> List[Dict[str, Any]]:
        pass

class Retrieval:
    retriever: Retriever
    reranker: Optional[Reranker] = None

    @classmethod
    def retrieve(cls, args: Dict[str, Any], ctx: Context) -> Context:
        items = cls.retriever.retrieve(ctx, **args)
        if cls.reranker:
            items = cls.reranker.rerank(items, ctx)
        ctx.working_memory.setdefault("retrieval", []).append(items)
        return ctx

class Generation:
    @classmethod
    def generate(cls, args: Dict[str, Any], ctx: Context) -> Context:
        text = GlobalRegistry.router.complete(prompt=args.get("prompt", ""))
        ctx.working_memory.setdefault("generations", []).append(text)
        return ctx

# ------------------------------
# Section 2/13: Model Router & LLM Client
# ------------------------------

class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        pass

class ModelRouter:
    def __init__(self, primary: ModelRef, fallback: List[ModelRef]):
        self.primary = primary
        self.fallback = fallback
        self._clients: Dict[str, LLMClient] = {}

    def register(self, model_name: str, client: LLMClient) -> None:
        self._clients[model_name] = client

    def complete(self, prompt: str, **kwargs) -> str:
        order = [self.primary.name] + [m.name for m in self.fallback]
        for m in order:
            client = self._clients.get(m)
            if not client:
                continue
            try:
                return client.complete(prompt, **kwargs)
            except Exception:
                continue
        raise RuntimeError("No model available")

# ------------------------------
# Section 10: Safety & Governance
# ------------------------------

class InputFilter(ABC):
    @abstractmethod
    def filter(self, msg: Message) -> Message:
        pass

class OutputFilter(ABC):
    @abstractmethod
    def filter(self, text: str) -> str:
        pass

class Safety:
    def __init__(self, inputs: List[InputFilter], outputs: List[OutputFilter]):
        self._in = inputs
        self._out = outputs

    def filter_in(self, msg: Message) -> Message:
        for f in self._in:
            msg = f.filter(msg)
        return msg

    def filter_out(self, text: str) -> str:
        for f in self._out:
            text = f.filter(text)
        return text

# ------------------------------
# Section 11: Telemetry, Evaluation & QA
# ------------------------------

class Tracer(ABC):
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass

class Telemetry:
    def __init__(self, tracer: Tracer, metrics: Callable[[str, Dict[str, Any]], None], logger: Callable[[str, Dict[str, Any]], None]):
        self._tracer = tracer
        self._metrics = metrics
        self._logger = logger

    def span(self, name: str) -> Tracer:
        return self._tracer

    def metric(self, name: str, tags: Dict[str, Any]):
        self._metrics(name, tags)

    def log(self, msg: str, data: Dict[str, Any]):
        self._logger(msg, data)

class Judge(ABC):
    @abstractmethod
    def score(self, ctx: Context, answer: str) -> float:
        pass

# ------------------------------
# Section 15: HITL & UX Hooks (minimal)
# ------------------------------

class HITLManager(ABC):
    @abstractmethod
    def should_escalate(self, ctx: Context) -> bool:
        pass

    @abstractmethod
    def escalate(self, ctx: Context) -> str:
        pass

# ------------------------------
# Section 8/12/17: The Agent Shell
# ------------------------------

class Agent:
    def __init__(self,
                 manifest: AgentManifest,
                 router: ModelRouter,
                 memory: MemorySystem,
                 tools: ToolRegistry,
                 safety: Safety,
                 telemetry: Telemetry,
                 planner: Planner,
                 critic: Critic,
                 orchestrator: Orchestrator,
                 hitl: Optional[HITLManager] = None):
        self.manifest = manifest
        self.router = router
        self.memory = memory
        self.tools = tools
        self.safety = safety
        self.telemetry = telemetry
        self.planner = planner
        self.critic = critic
        self.orchestrator = orchestrator
        self.hitl = hitl

    def handle(self, user_msg: Message) -> str:
        with self.telemetry.span("ingress"):
            msg = self.safety.filter_in(user_msg)
            ctx = self.memory.context(msg)
            ctx.conversation.append(msg)

        plan = self.planner.make_plan(msg, ctx, self.router)
        ctx = self.orchestrator.run(ctx, plan)

        # Synthesize answer from working memory (toy)
        generations = ctx.working_memory.get("generations", [])
        answer = generations[-1] if generations else "(no answer)"

        ok, reason = self.critic.critique(ctx)
        if not ok:
            ctx.error_state["critic"] = reason

        if self.hitl and self.hitl.should_escalate(ctx):
            answer = self.hitl.escalate(ctx)

        answer = self.safety.filter_out(answer)
        self.telemetry.metric("answer_ready", {"len": len(answer)})
        return answer

# ------------------------------
# Section 9: Example Concrete Pieces (stubs)
# ------------------------------

class SimplePlanner(Planner):
    def make_plan(self, msg: Message, ctx: Context, router: ModelRouter) -> Plan:
        steps = [
            PlanStep(id="p1", action="retrieve", args={"q": msg.content}),
            PlanStep(id="p2", action="generate", args={"prompt": f"Answer: {msg.content}"}),
            PlanStep(id="p3", action="verify")
        ]
        return Plan(steps=steps)

class PassFailCritic(Critic):
    def critique(self, ctx: Context) -> Tuple[bool, Optional[str]]:
        # Example: require at least one generation
        has_gen = bool(ctx.working_memory.get("generations"))
        return (has_gen, None if has_gen else "No generation produced")

class NoopWorkingMemory(WorkingMemory):
    def context(self, msg: Message) -> Context:
        return Context(conversation=[])

class NoopEpisodic(EpisodicMemory):
    def record(self, event: Dict[str, Any]) -> None:
        pass

class NoopSemantic(SemanticMemory):
    def retrieve(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        return []

class NoopProcedural(ProceduralMemory):
    def write_playbook(self, entry: Dict[str, Any]) -> None:
        pass
    def recall(self, query: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
        return []

class HybridRetrieverImpl(Retriever):
    def retrieve(self, ctx: Context, **kwargs) -> Dict[str, Any]:
        # Consult memory.semantic + web/API as needed in real impl
        return {"items": []}

class IdentityReranker(Reranker):
    def rerank(self, items: List[Dict[str, Any]], ctx: Context) -> List[Dict[str, Any]]:
        return items

class SimpleLLM(LLMClient):
    def complete(self, prompt: str, **kwargs) -> str:
        return f"[LLM completion for prompt: {prompt[:64]}...]"

class SimpleInputFilter(InputFilter):
    def filter(self, msg: Message) -> Message:
        return msg

class SimpleOutputFilter(OutputFilter):
    def filter(self, text: str) -> str:
        return text

class BasicHITL(HITLManager):
    def should_escalate(self, ctx: Context) -> bool:
        return "critic" in ctx.error_state
    def escalate(self, ctx: Context) -> str:
        return "(Draft sent to human for review)"

# Wire global facilities used by class-level facades
class GlobalRegistry:
    critic: Critic
    router: ModelRouter

# Plug default retrieval/reranking impls into facades
Retrieval.retriever = HybridRetrieverImpl()
Retrieval.reranker = IdentityReranker()

# ------------------------------
# Section 17: Minimal Usage Example
# ------------------------------

def build_minimal_agent() -> Agent:
    manifest = AgentManifest(
        name="KPIAnalyst",
        persona_description="Analytical, precise, concise",
        audience=["analyst", "manager"],
        objectives_primary=["answer KPI questions", "diagnostics"],
        constraints=["<=6s p95", "no PII exfiltration"],
        policies=PolicyConfig(tool_use="prefer data tools", hitl_on=["low confidence"]),
        routing=RoutingConfig(),
        input_schema=JsonSchemaRef("schemas/input.json"),
        output_schema=JsonSchemaRef("schemas/output.json"),
        redaction_rules=["mask_email"],
        models_primary=ModelRef(provider="local", name="gpt-large"),
        models_fallback=[ModelRef(provider="local", name="gpt-medium")],
        embeddings=None,
        memory=MemoryConfig(),
        tools=[ToolSpec(id="kpi.fetch", description="Fetch KPI", input_schema=JsonSchemaRef("tools/kpi.fetch.input.json"), output_schema=JsonSchemaRef("tools/kpi.fetch.output.json"))],
        safety=SafetyConfig(content_policies=["no_medical"], pii_blocklist=["ssn", "aadhaar"]),
        telemetry=TelemetryConfig(),
        deployment=DeploymentConfig()
    )

    # Router & models
    router = ModelRouter(manifest.models_primary, manifest.models_fallback)
    router.register("gpt-large", SimpleLLM())
    router.register("gpt-medium", SimpleLLM())

    # Memory
    memory = MemorySystem(working=NoopWorkingMemory(), episodic=NoopEpisodic(), semantic=NoopSemantic(), procedural=NoopProcedural())

    # Tools registry (register real tools in your app)
    tools = ToolRegistry()

    # Safety & telemetry
    safety = Safety(inputs=[SimpleInputFilter()], outputs=[SimpleOutputFilter()])
    telemetry = Telemetry(tracer=Tracer(), metrics=lambda n, t: None, logger=lambda m, d: None)

    # Planner, critic, orchestrator, HITL
    planner = SimplePlanner()
    critic = PassFailCritic()
    orchestrator = FSMOrchestrator()
    hitl = BasicHITL()

    agent = Agent(
        manifest=manifest,
        router=router,
        memory=memory,
        tools=tools,
        safety=safety,
        telemetry=telemetry,
        planner=planner,
        critic=critic,
        orchestrator=orchestrator,
        hitl=hitl
    )

    # Expose global for facades
    GlobalRegistry.critic = critic
    GlobalRegistry.router = router

    return agent

if __name__ == "__main__":
    agent = build_minimal_agent()
    msg = Message(id="1", ts="2025-09-10T10:00:00+05:30", role="user", content="Show revenue contribution by channel.")
    print(agent.handle(msg))
