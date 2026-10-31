import os
import httpx
from typing import Dict, Any, Optional, List

def get_site_url() -> str:
    """Returns the configured production frontend origin for OpenRouter HTTP-Referer header."""
    return os.getenv("FRONTEND_URL") or os.getenv("SITE_URL") or "http://localhost:5173"

def normalize_base_url(base_url: Optional[str], provider: str) -> str:
    prov = (provider or "openai-compatible").lower()
    url = (base_url or "").strip()
    
    while url.endswith("/"):
        url = url[:-1]

    # OpenRouter or openrouter.ai in URL
    if prov == "openrouter" or "openrouter.ai" in url:
        if not url:
            return "https://openrouter.ai/api/v1"
        if url == "https://openrouter.ai":
            return "https://openrouter.ai/api/v1"
        if url.endswith("/api/v1"):
            return url
        return f"{url}/api/v1"

    if prov == "openai":
        if not url:
            return "https://api.openai.com/v1"
        return url

    if not url:
        return "https://openrouter.ai/api/v1"

    return url


def test_ai_connection(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Minimal safe API request using selected provider & model to test connection.
    Does not persist or log API key.
    """
    key_clean = (api_key or "").strip()
    if not key_clean:
        return {"success": False, "message": "API key cannot be empty."}

    provider_clean = (provider or "openai-compatible").lower()
    norm_base = normalize_base_url(base_url, provider_clean)

    model_clean = (model or "").strip()
    if not model_clean:
        if "openrouter" in provider_clean or "openrouter.ai" in norm_base:
            model_clean = "openai/gpt-4o-mini"
        elif provider_clean in ("gemini", "google"):
            model_clean = "gemini-1.5-flash"
        else:
            model_clean = "gpt-4o-mini"

    try:
        with httpx.Client(timeout=15.0) as client:
            if provider_clean in ("gemini", "google") and "generativelanguage.googleapis.com" not in (base_url or ""):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_clean}:generateContent?key={key_clean}"
                response = client.post(
                    url,
                    json={
                        "contents": [{"parts": [{"text": "Respond with exactly: Connection successful."}]}],
                        "generationConfig": {"maxOutputTokens": 10}
                    }
                )
                if response.status_code == 200:
                    return {"success": True, "message": "✓ Connection Successful! Configuration Saved."}
                elif response.status_code in (401, 403):
                    return {"success": False, "message": "✕ Invalid API key or access denied by provider."}
                elif response.status_code == 404:
                    return {"success": False, "message": f"✕ Selected model '{model_clean}' was not found."}
                elif response.status_code == 429:
                    return {"success": False, "message": "✕ Rate limit reached. Please try again later."}
                else:
                    return {"success": False, "message": f"✕ Provider request failed with status code {response.status_code}."}

            else:
                chat_url = f"{norm_base}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {key_clean}",
                    "Content-Type": "application/json"
                }
                if "openrouter.ai" in norm_base:
                    headers["HTTP-Referer"] = get_site_url()
                    headers["X-Title"] = "Conflict & View Serializability Analyzer"

                payload = {
                    "model": model_clean,
                    "messages": [
                        {"role": "user", "content": "Respond with exactly: Connection successful."}
                    ],
                    "max_tokens": 10
                }
                response = client.post(chat_url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data or "id" in data or "object" in data:
                        return {"success": True, "message": "✓ Connection Successful! Configuration Saved."}
                    return {"success": True, "message": "✓ Connection Successful! Configuration Saved."}
                elif response.status_code in (401, 403):
                    if "openrouter" in provider_clean or "openrouter.ai" in norm_base:
                        return {"success": False, "message": "✕ Invalid OpenRouter API key."}
                    return {"success": False, "message": "✕ Invalid API key."}
                elif response.status_code == 404:
                    return {"success": False, "message": f"✕ Model '{model_clean}' or API endpoint not found."}
                elif response.status_code == 429:
                    return {"success": False, "message": "✕ Rate limit reached. Please try again later."}
                else:
                    err_msg = ""
                    try:
                        err_data = response.json()
                        if "error" in err_data:
                            if isinstance(err_data["error"], dict):
                                err_msg = err_data["error"].get("message", "")
                            elif isinstance(err_data["error"], str):
                                err_msg = err_data["error"]
                    except Exception:
                        pass
                    msg = f"✕ Connection failed: {err_msg}" if err_msg else f"✕ Connection failed with HTTP {response.status_code}."
                    return {"success": False, "message": msg}

    except httpx.TimeoutException:
        return {"success": False, "message": "✕ Connection timed out. Please verify Base URL and network connectivity."}
    except Exception:
        return {"success": False, "message": "✕ Unable to reach the AI provider. Please verify Base URL and settings."}



def discover_ai_models(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Attempt GET <BASE_URL>/models using Bearer API key.
    """
    key_clean = (api_key or "").strip()
    if not key_clean:
        return {"success": False, "message": "API key required to discover models.", "models": []}

    provider_clean = (provider or "openai-compatible").lower()

    try:
        with httpx.Client(timeout=15.0) as client:
            if provider_clean in ("gemini", "google") and "generativelanguage.googleapis.com" not in (base_url or ""):
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key_clean}"
                res = client.get(url)
                if res.status_code == 200:
                    raw_models = res.json().get("models", [])
                    models_list = []
                    for m in raw_models:
                        name = m.get("name", "").replace("models/", "")
                        if "gemini" in name:
                            models_list.append({"id": name, "name": m.get("displayName", name)})
                    return {"success": True, "models": models_list}
                return {"success": False, "message": "Unable to discover models. You can enter a model ID manually.", "models": []}
            else:
                norm_base = normalize_base_url(base_url, provider_clean)
                models_url = f"{norm_base}/models"
                headers = {
                    "Authorization": f"Bearer {key_clean}",
                    "Content-Type": "application/json"
                }
                if "openrouter.ai" in norm_base:
                    headers["HTTP-Referer"] = get_site_url()
                    headers["X-Title"] = "Conflict & View Serializability Analyzer"

                res = client.get(models_url, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    items = data.get("data", [])
                    if isinstance(items, list):
                        models_list = []
                        for item in items:
                            if isinstance(item, dict) and "id" in item:
                                m_id = item["id"]
                                m_name = item.get("name", m_id)
                                models_list.append({"id": m_id, "name": m_name})
                        models_list.sort(key=lambda x: x["id"])
                        return {"success": True, "models": models_list}
                return {"success": False, "message": "Unable to discover models. You can enter a model ID manually.", "models": []}
    except Exception:
        return {"success": False, "message": "Unable to discover models. You can enter a model ID manually.", "models": []}


def generate_ai_explanation(
    analysis_data: Optional[Dict[str, Any]] = None,
    prompt_type: str = "explain_result",
    custom_question: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate generic AI response or contextual DBMS tutoring using configured provider.
    Supports arbitrary user questions and conversation history.
    """
    effective_key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()

    if not effective_key:
        return {
            "success": False,
            "prompt_type": prompt_type,
            "explanation": "Bring Your Own Key to enable AI explanations.",
            "is_ai_generated": False,
            "error_code": "NO_KEY"
        }

    provider_clean = (provider or ("openrouter" if effective_key.startswith("sk-or-") else ("openai" if effective_key.startswith("sk-") else "openai-compatible"))).lower()
    norm_base = normalize_base_url(base_url, provider_clean)

    selected_model = (model or "").strip()
    if not selected_model:
        if "openrouter" in provider_clean or "openrouter.ai" in norm_base:
            selected_model = "openai/gpt-4o-mini"
        elif provider_clean in ("gemini", "google"):
            selected_model = "gemini-1.5-flash"
        else:
            selected_model = "gpt-4o-mini"

    # Base System Prompt
    system_prompt = """You are an expert AI Tutor and computer science professor.
You provide clear, accurate, and helpful answers to any user questions.

If the user asks about database management systems (DBMS), transaction processing, or schedule serializability:
- Provide clear, educational explanations.
- If authoritative schedule analysis context is supplied below, strictly adhere to those computed facts. Do NOT recalculate, contradict, or alter backend algorithmic results.

If the user asks general questions (e.g., C++, Java, networking, general knowledge, math, greetings):
- Answer normally, clearly, and accurately without requiring any DBMS context."""

    # Append DBMS analysis context if available
    if analysis_data and isinstance(analysis_data, dict) and analysis_data.get("schedule_text"):
        combined = analysis_data.get("combined_result", {})
        conflict_serializable = combined.get("conflict_serializable", False)
        view_serializable = combined.get("view_serializable", False)
        schedule = analysis_data.get("schedule_text", "")
        conflicts = analysis_data.get("conflicts", [])
        cycles = analysis_data.get("formatted_cycles", [])
        cycles_str = ", ".join(cycles) if cycles else "None"
        view_info = analysis_data.get("view_analysis", {})
        eq_orders = view_info.get("equivalent_orders", [])
        eq_orders_str = ", ".join(eq_orders) if eq_orders else "None"
        top_orders = analysis_data.get("formatted_topological_orders", [])
        top_orders_str = ", ".join(top_orders) if top_orders else "None"
        data_items_str = ", ".join(analysis_data.get("data_items", []))

        system_prompt += f"""

[Current Active DBMS Schedule Context]
Schedule: {schedule}
Data Items: {data_items_str}
Conflict Serializable: {conflict_serializable}
View Serializable: {view_serializable}
Precedence Graph Cycles: {cycles_str}
Topological Orders (Conflict): {top_orders_str}
Equivalent Serial Orders (View): {eq_orders_str}
Conflicts Breakdown: {conflicts}
"""

    api_messages = [{"role": "system", "content": system_prompt}]

    # Build conversation payload
    if messages and isinstance(messages, list) and len(messages) > 0:
        for m in messages:
            if isinstance(m, dict) and "role" in m and "content" in m:
                if m["role"] in ("user", "assistant"):
                    api_messages.append({"role": m["role"], "content": m["content"]})
    else:
        if custom_question and custom_question.strip():
            user_text = custom_question.strip()
        elif prompt_type == "beginner":
            user_text = "Explain the current schedule analysis like I am a complete beginner (ELI5)."
        elif prompt_type == "conflicts":
            user_text = "Explain all conflict operations and precedence graph edges in detail."
        elif prompt_type == "viva":
            user_text = "Generate 3 viva interview questions with answers based on this schedule."
        else:
            user_text = "Explain the current schedule analysis in detail."
            
        api_messages.append({"role": "user", "content": user_text})

    try:
        with httpx.Client(timeout=30.0) as client:
            if provider_clean in ("gemini", "google") and "generativelanguage.googleapis.com" not in norm_base:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={effective_key}"
                gemini_contents = []
                for m in api_messages:
                    if m["role"] == "system":
                        continue
                    role_name = "user" if m["role"] == "user" else "model"
                    gemini_contents.append({"role": role_name, "parts": [{"text": m["content"]}]})
                if not gemini_contents:
                    gemini_contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

                payload = {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": gemini_contents
                }
                response = client.post(url, headers={"Content-Type": "application/json"}, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        content = "".join(p.get("text", "") for p in parts)
                        return {
                            "success": True,
                            "prompt_type": prompt_type,
                            "explanation": content,
                            "is_ai_generated": True,
                            "source": f"Google Gemini ({selected_model})"
                        }
                    return {"success": False, "explanation": "Received empty response from AI provider.", "error_code": "EMPTY_RESPONSE"}
                elif response.status_code == 429:
                    return {"success": False, "explanation": "Rate limit reached. Please try again later.", "error_code": "RATE_LIMIT"}
                elif response.status_code in (401, 403, 400):
                    return {"success": False, "explanation": "Invalid API key or model. Please verify your settings.", "error_code": "INVALID_KEY"}
                else:
                    return {"success": False, "explanation": "AI provider request failed. Please verify your settings.", "error_code": "PROVIDER_ERROR"}
            else:
                chat_url = f"{norm_base}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {effective_key}",
                    "Content-Type": "application/json"
                }
                if "openrouter.ai" in norm_base:
                    headers["HTTP-Referer"] = get_site_url()
                    headers["X-Title"] = "Conflict & View Serializability Analyzer"

                payload = {
                    "model": selected_model,
                    "messages": api_messages,
                    "temperature": 0.3
                }
                response = client.post(chat_url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        msg = choices[0].get("message", {})
                        content = msg.get("content", "")
                        return {
                            "success": True,
                            "prompt_type": prompt_type,
                            "explanation": content,
                            "is_ai_generated": True,
                            "source": f"AI Assistant ({selected_model})"
                        }
                    return {"success": False, "explanation": "Provider returned an unexpected response format.", "error_code": "MALFORMED_RESPONSE"}
                elif response.status_code in (401, 403):
                    if "openrouter" in provider_clean or "openrouter.ai" in norm_base:
                        return {"success": False, "explanation": "Invalid OpenRouter API key.", "error_code": "INVALID_KEY"}
                    return {"success": False, "explanation": "Invalid API key.", "error_code": "INVALID_KEY"}
                elif response.status_code == 404:
                    return {"success": False, "explanation": "API endpoint or model not found.", "error_code": "NOT_FOUND"}
                elif response.status_code == 429:
                    return {"success": False, "explanation": "Rate limit reached. Please try again later.", "error_code": "RATE_LIMIT"}
                else:
                    err_msg = ""
                    try:
                        err_data = response.json()
                        if "error" in err_data:
                            if isinstance(err_data["error"], dict):
                                err_msg = err_data["error"].get("message", "")
                            elif isinstance(err_data["error"], str):
                                err_msg = err_data["error"]
                    except Exception:
                        pass
                    explanation = err_msg if err_msg else "AI request failed. Please verify your settings."
                    return {"success": False, "explanation": explanation, "error_code": "PROVIDER_ERROR"}

    except httpx.TimeoutException:
        return {"success": False, "explanation": "Request timed out while connecting to AI provider.", "error_code": "TIMEOUT"}
    except Exception:
        return {"success": False, "explanation": "Unable to reach the AI provider. Check your network connection.", "error_code": "NETWORK_ERROR"}
