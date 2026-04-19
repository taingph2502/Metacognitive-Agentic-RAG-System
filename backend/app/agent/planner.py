import logging

from langchain_core.messages import HumanMessage

from app.llm import ainvoke
from app.text_utils import extract_json_object

logger = logging.getLogger(__name__)


def classify_rule_based(query: str) -> str:
    """Lightweight rule-based fallback."""
    q = query.lower()
    if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast", "better than", "which is"]):
        return "complex"
    if any(w in q for w in ["why", "how does", "explain", "what caused", "what led", "relationship between", "connection between"]):
        return "multi-hop"
    return "simple"


PLAN_PROMPT = """\
Bạn là một AI RAG Router. Hãy phân loại truy vấn sau thành một trong ba độ khó:
- "simple": Câu hỏi thực tế đơn giản, tra cứu 1 lần là có kết quả.
- "complex": Câu hỏi cần tổng hợp nhiều thông tin hoặc có đại từ nhân xưng cần làm rõ ngữ cảnh.
- "multi-hop": Câu hỏi suy luận chéo, so sánh, cần đi qua nhiều bước tài liệu mới trả lời được.

Truy vấn: {query}

Phản hồi dưới dạng JSON hợp lệ chứa 1 trường duy nhất:
- "complexity": "simple", "complex", hoặc "multi-hop"

Chỉ trả về JSON, không kèm giải thích."""


async def plan_query(query: str) -> dict:
    """Returns {"complexity": str}"""
    prompt = PLAN_PROMPT.format(query=query)
    try:
        response = await ainvoke([HumanMessage(content=prompt)], call_site="planner")
        result = extract_json_object(response.content.strip())
        if result:
            comp = result.get("complexity")
            if comp not in ("simple", "complex", "multi-hop"):
                result["complexity"] = classify_rule_based(query)
            return result
    except Exception as e:
        logger.warning(f"Failed to parse planner LLM output: {e}", exc_info=True)

    complexity = classify_rule_based(query)
    return {"complexity": complexity}
