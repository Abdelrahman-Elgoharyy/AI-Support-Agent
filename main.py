"""
Customer Support AI Agent — Complete Production Code
================================================    
"""

import logging
import uuid
from typing import Dict

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.hooks import (
    AfterInvocationEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from strands.models import BedrockModel
import boto3
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser

from mcp.client.streamable_http import streamable_http_client
from strands.tools.mcp.mcp_client import MCPClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CSAI_Agent")

app = BedrockAgentCoreApp()
model = BedrockModel(model_id="us.amazon.nova-2-lite-v1:0")

GATEWAY_URL = "https://customersupportgateway-dhsztyf6p3.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "EZJYNINDNS"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-Ndm301HHOi"

memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ===========================================================================
# NAMESPACE HELPER
# ===========================================================================

def get_namespaces(mem_client, memory_id: str) -> Dict[str, str]:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    return {strategy["type"]: strategy["namespaces"][0] for strategy in strategies}


# ===========================================================================
# MEMORY HOOK
# ===========================================================================

class WanderBotMemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(self, memory_client, memory_id: str):
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(self.memory_client, self.memory_id)
        logger.info("Namespaces loaded: %s", self.namespaces)

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self._retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self._save_interaction)

    def _retrieve_customer_context(self, event) -> None:
        """Retrieve relevant memories and prepend them to the user message."""
        actor_id = event.agent.state.get("actor_id")
        if not actor_id:
            return

        messages = event.agent.messages
        if not messages or messages[-1]["role"] != "user":
            return

        content = messages[-1].get("content", [])
        if not content or not isinstance(content, list):
            return

        user_query = ""
        for block in content:
            if isinstance(block, dict) and "text" in block:
                user_query = block["text"]
                break

        if not user_query:
            return

        retrieved_memories = []
        for strat_type, ns_template in self.namespaces.items():
            namespace = ns_template.format(actorId=actor_id)
            try:
                response = self.memory_client.retrieve_memories(
                    self.memory_id, namespace, user_query, top_k=5
                )
                # Handle different response structures gracefully
                summaries = []
                if isinstance(response, dict):
                    summaries = response.get("memorySummaries", response.get("memories", []))
                elif isinstance(response, list):
                    summaries = response

                for mem in summaries:
                    mem_text = ""
                    if isinstance(mem, dict):
                        # Check all common keys where memory text might reside
                        mem_text = (
                            mem.get("content", {}).get("text", "") or
                            mem.get("memoryContent", {}).get("text", "") or
                            mem.get("text", "")
                        )
                        if not mem_text and "content" in mem and isinstance(mem["content"], str):
                            mem_text = mem["content"]
                    elif hasattr(mem, "get"):
                        mem_text = mem.get("text", "")

                    if mem_text:
                        retrieved_memories.append(f"[{strat_type}] {mem_text.strip()}")
            except Exception as e:
                logger.warning(f"Failed to retrieve memory for {strat_type}: {e}")

        if retrieved_memories:
            memories_block = "\n".join(retrieved_memories)
            enhanced_text = f"Customer Context:\n{memories_block}\n\n{user_query}"
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    block["text"] = enhanced_text
                    break

    def _save_interaction(self, event) -> None:
        """Save the completed turn to memory after the agent responds."""
        actor_id = event.agent.state.get("actor_id")
        session_id = event.agent.state.get("session_id")
        if not actor_id or not session_id:
            return

        messages = event.agent.messages
        user_query = None
        assistant_response = None

        for msg in reversed(messages):
            role = msg.get("role")
            content = msg.get("content", [])
            text = ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        text = block["text"]
                        break
            elif isinstance(content, str):
                text = content

            if role == "assistant" and not assistant_response and text:
                assistant_response = text
            elif role == "user" and not user_query and text:
                if "Customer Context:" in text:
                    parts = text.split("\n\n")
                    if len(parts) > 1:
                        text = parts[-1]
                user_query = text

            if user_query and assistant_response:
                break

        if user_query and assistant_response:
            try:
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    messages=[
                        (user_query, "USER"),
                        (assistant_response, "ASSISTANT"),
                    ]
                )
                logger.info("Saved interaction to memory for actor %s", actor_id)
            except Exception as e:
                logger.warning(f"Failed to save interaction to memory: {e}")


# ===========================================================================
# TOOLS
# ===========================================================================

@tool
def search_knowledge_base(query: str) -> str:
    """Search the Amazon product catalog and support knowledge base."""
    if not KB_ID:
        return "Knowledge base not configured."
    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query}
        )
        results = response.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."
        chunks = [res.get("content", {}).get("text", "") for res in results if res.get("content", {}).get("text")]
        return "\n---\n".join(chunks) if chunks else "No text content found."
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        return f"Error searching knowledge base: {e}"


@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """Calculate the loyalty discount for a customer order using the AgentCore Code Interpreter."""
    import json
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

order_total = float({order_total})
loyalty_points = int({loyalty_points})
tier = "{tier}"
category = "{product_category}"

max_points_allowed = int((order_total * 0.50) * 100)
usable_points = min(loyalty_points, max_points_allowed)
points_redeemed = (usable_points // 500) * 500
points_discount = points_redeemed / 100.0

subtotal_after_points = max(0.0, order_total - points_discount)

tier_rate = tier_rates.get(tier, 0.0)
tier_discount_amount = subtotal_after_points * tier_rate
final_total = max(0.0, subtotal_after_points - tier_discount_amount)

total_savings = points_discount + tier_discount_amount
remaining_points = loyalty_points - points_redeemed

earn_rate = earn_rates.get(category, 1)
points_earned = int(final_total * earn_rate)
remaining_points += points_earned

result = {{
    "order_total": round(order_total, 2),
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier": tier,
    "tier_discount_amount": round(tier_discount_amount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points
}}
print(json.dumps(result))
"""
    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke("executeCode", {
                "code": code,
                "language": "python",
                "clearContext": True,
            })
        
        output_str = ""
        stream = response.get("stream", []) if isinstance(response, dict) else response
        for event in stream:
            if isinstance(event, dict):
                if "output" in event:
                    output_str += event["output"]
                elif "result" in event:
                    res = event["result"]
                    if isinstance(res, dict):
                        return json.dumps(res)
                    elif isinstance(res, str):
                        return res
            elif hasattr(event, "get"):
                if event.get("output"):
                    output_str += event.get("output")
        
        if output_str.strip():
            return output_str.strip()
            
        return str(response)

    except Exception as e:
        logger.warning(f"Code Interpreter failed, using local tier-only fallback: {e}")
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_rate = tier_rates.get(tier, 0.0)
        tier_disc = order_total * tier_rate
        final = max(0.0, order_total - tier_disc)
        
        # Tier-only fallback response as required by mentor
        return json.dumps({
            "order_total": round(order_total, 2),
            "tier": tier,
            "tier_discount_amount": round(tier_disc, 2),
            "final_total": round(final, 2)
        })


# ===========================================================================
# ENTRY POINT
# ===========================================================================

SYSTEM_PROMPT = (
    "You are an expert, friendly, and professional AI customer support assistant for an e-commerce platform. "
    "You help customers track orders, process returns/refunds, look up product specifications and store policies "
    "via the knowledge base, calculate exact loyalty discounts using the code sandbox, and browse the web for live info. "
    "Always maintain a helpful and polite tone. Use the available tools when needed to give accurate, grounded answers."
)


@app.entrypoint
async def invoke(payload: dict, context=None) -> dict:
    """Main handler called by AgentCore for every incoming request."""
    user_message = payload.get("prompt", payload.get("message", "Hello!"))
    actor_id = payload.get("customer_id", payload.get("actor_id", "default_customer"))
    session_id = payload.get("session_id", str(uuid.uuid4()))

    logger.info("Session %s | Actor %s | User: %s", session_id, actor_id, user_message[:80])

    memory_hook = WanderBotMemoryHook(memory_client=memory_client, memory_id=MEMORY_ID)
    browser_tool = AgentCoreBrowser(region=REGION)

    tools = [
        search_knowledge_base,
        calculate_loyalty_discount,
        browser_tool.browser,
    ]

    client = MCPClient(
        lambda: streamable_http_client(url=GATEWAY_URL)
    )

    # 1. Open the gateway client context once
    with client as gateway_client:
        try:
            gateway_tools = gateway_client.list_tools_sync()
            tools.extend(gateway_tools)
            logger.info(
                "Gateway connected successfully. Loaded %d tools.",
                len(gateway_tools),
            )
        except TimeoutError:
            logger.exception("Gateway tool loading timed out")
        except ConnectionError:
            logger.exception("Gateway connection failed")
        except Exception as exc:
            logger.exception(
                "Gateway tool loading failed: %s", exc
            )

        # 2. Initialize and run the agent ONCE right here inside the active session
        agent = Agent(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            state={"session_id": session_id, "actor_id": actor_id},
            hooks=[memory_hook],
        )

        return agent(user_message)


if __name__ == "__main__":
    app.run()