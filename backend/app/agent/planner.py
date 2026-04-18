from langchain_core.messages import HumanMessage

from app.llm import ainvoke
from app.text_utils import extract_json_object


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

Phản hồi dưới dạng JSON hợp lệ chứa 2 trường sau:
- "complexity": "simple", "complex", hoặc "multi-hop"
- "rewritten_query": Nếu complexity là "simple", trả về chuỗi rỗng "". Nếu là "complex" hoặc "multi-hop", hãy viết lại thành MỘT câu hỏi duy nhất, rõ ngữ cảnh và đầy đủ chủ ngữ. QUAN TRỌNG: TUYỆT ĐỐI GIỮ NGUYÊN danh từ riêng, mã lỗi, tên sản phẩm hoặc thuật ngữ kỹ thuật, không được dịch hay loại bỏ.

Chỉ trả về JSON, không kèm giải thích."""


async def plan_query(query: str) -> dict:
    """Returns {"complexity": str, "rewritten_query": str}"""
    prompt = PLAN_PROMPT.format(query=query)
    try:
        response = await ainvoke([HumanMessage(content=prompt)], call_site="planner")
        result = extract_json_object(response.content.strip())
        if result:
            comp = result.get("complexity")
            if comp not in ("simple", "complex", "multi-hop"):
                result["complexity"] = classify_rule_based(query)
            if "rewritten_query" not in result:
                result["rewritten_query"] = "" if result["complexity"] == "simple" else query
            return result
    except Exception:
        pass

    complexity = classify_rule_based(query)
    return {"complexity": complexity, "rewritten_query": "" if complexity == "simple" else query}
