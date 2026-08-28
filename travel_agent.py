"""使用大模型规划行程、使用高德地图核验路线的自驾路线 Agent。"""

import argparse
import json
import re
import sys
from typing import Any

from gaode import driving_route, geocode, geocode_candidates, organize_routes
from llm import call_llm


PLANNER_SYSTEM_PROMPT = """
你是一名中国境内自驾路线规划师。请根据用户的起点和要求，设计合理的逐日路线。
你只负责决定天数、每日起终点、途中停靠点和游览安排；里程和驾驶时间将由地图接口计算。
地点名称必须具体到高德地图能检索的城市、区县、景区或建筑名称。每天必须从上一天终点出发，
第一天必须从用户给出的起点出发。不要凭空增加用户明确排除的地点。

只输出合法 JSON，不要输出 Markdown 代码块或解释，格式必须为：
{
  "trip_title": "行程名称",
  "days": [
    {
      "day": 1,
      "start": "当天起点",
      "end": "当天终点/住宿地",
      "waypoints": ["按驾车顺序排列的中途停靠点"],
      "activities": ["当天建议活动"],
      "notes": "规划理由或注意事项"
    }
  ]
}
waypoints 不包含 start 和 end，没有中途点时使用空数组。
""".strip()

REVISER_SYSTEM_PROMPT = """
你是一名中国境内自驾路线规划师。用户会基于上一版行程提出修改意见，请完整理解修改要求，
在保留未被要求改变的安排的同时生成一份完整的新行程。地图数据中的里程和时间仅供理解上一版路线，
新路线的实际里程和时间会由高德地图重新计算，不要把它们写入 JSON。
每天必须从上一天终点出发，第一天必须从固定起始点出发。地点名称必须具体到高德地图能够检索。

只输出合法 JSON，不要输出 Markdown 代码块或解释，格式必须为：
{
  "trip_title": "行程名称",
  "days": [
    {
      "day": 1,
      "start": "当天起点",
      "end": "当天终点/住宿地",
      "waypoints": ["按驾车顺序排列的中途停靠点"],
      "activities": ["当天建议活动"],
      "notes": "规划理由或注意事项"
    }
  ]
}
waypoints 不包含 start 和 end，没有中途点时使用空数组。
""".strip()


FINAL_SYSTEM_PROMPT = """
你是一名严谨的自驾旅行顾问。请把规划草案和高德地图核验数据整理成中文详细行程。
必须忠实使用地图数据中的天数、起终点、逐段路线、总里程和预计驾驶时长，不得修改或虚构这些数字。
可以补充合理的游览节奏、用餐、住宿区域、安全提醒，但不确定的信息要明确标注“建议提前确认”。
使用清晰的 Markdown 输出：先给行程概览，再逐日列出起点、终点、途经点、分段路线、当日总里程、
预计驾驶时长、活动安排和注意事项。时间使用“约 X 小时 Y 分钟”，里程保留 1 位小数。
""".strip()


def _parse_json(text: str) -> dict[str, Any]:
    """从模型文本中提取 JSON 对象。"""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"大模型未返回有效 JSON：{text}")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("路线规划结果必须是 JSON 对象")
    return value


def _validate_plan(plan: dict[str, Any], origin: str) -> None:
    days = plan.get("days")
    if not isinstance(days, list) or not days:
        raise ValueError("大模型没有生成有效的逐日行程")

    previous_end = origin
    for index, day in enumerate(days, start=1):
        if not isinstance(day, dict) or not day.get("start") or not day.get("end"):
            raise ValueError(f"第 {index} 天缺少起点或终点")
        if day["start"].strip() != previous_end.strip():
            # 统一相邻两天地点文字，避免同一地点的轻微表述差异破坏路线衔接。
            day["start"] = previous_end
        if not isinstance(day.get("waypoints", []), list):
            raise ValueError(f"第 {index} 天 waypoints 必须是列表")
        day["day"] = index
        day.setdefault("activities", [])
        day.setdefault("notes", "")
        previous_end = day["end"]


def create_draft(
    origin: str,
    requirements: str,
    previous_trip: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """让大模型将用户要求拆解为结构化逐日行程。"""
    if previous_trip is None:
        system_prompt = PLANNER_SYSTEM_PROMPT
        prompt = f"起始点：{origin}\n具体要求：{requirements}"
    else:
        system_prompt = REVISER_SYSTEM_PROMPT
        prompt = (
            f"固定起始点：{origin}\n"
            f"用户本轮修改要求：{requirements}\n\n"
            "上一版已经高德核验的完整行程：\n"
            f"{json.dumps(previous_trip, ensure_ascii=False, indent=2)}"
        )
    last_error: Exception | None = None
    for attempt in range(2):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if attempt and last_error:
            messages.append(
                {
                    "role": "user",
                    "content": f"上次输出无法解析，错误是：{last_error}。请严格按指定 JSON 重新输出。",
                }
            )
        try:
            plan = _parse_json(
                call_llm(
                    messages,
                    temperature=0.5,
                    reasoning_effort="low",
                    timeout=120,
                )
            )
            _validate_plan(plan, origin)
            return plan
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"大模型连续两次未生成有效路线：{last_error}")


def _route_leg(
    start: str,
    end: str,
    start_location: str | None = None,
) -> dict[str, Any]:
    """用高德查询一个地点到另一个地点的首选驾车路线。"""
    try:
        start_location = start_location or geocode(start)
        is_administrative = bool(
            re.search(r"(?:省|市|自治州|地区|盟|县|区|旗|镇|乡)$", end.strip())
        )
        end_locations = (
            [geocode(end)]
            if is_administrative
            else geocode_candidates(end, reference_location=start_location)
        )
        route_options = []
        for end_location in end_locations:
            routes = organize_routes(driving_route(start_location, end_location))
            if routes:
                route_options.append((routes[0], end_location))
    except Exception as exc:
        raise RuntimeError(f"高德地图核验路段失败（{start} -> {end}）：{exc}") from exc
    if not route_options:
        raise ValueError(f"高德地图没有找到驾车路线：{start} -> {end}")

    # 同名 POI 可能分布在不同省份，选择从上一站驾车距离最短的候选。
    route, end_location = min(
        route_options,
        key=lambda item: item[0]["summary"]["distance_m"],
    )
    summary = route["summary"]
    road_names: list[str] = []
    for step in route["steps"]:
        road = step.get("road", "").strip()
        if road and road not in road_names:
            road_names.append(road)
    return {
        "start": start,
        "end": end,
        "distance_km": round(summary["distance_m"] / 1000, 1),
        "duration_minutes": round(summary["duration_s"] / 60),
        "tolls_yuan": summary["tolls"],
        "main_roads": road_names[:8],
        "_end_location": end_location,
    }


def verify_routes(plan: dict[str, Any]) -> dict[str, Any]:
    """逐日逐段查询高德，并计算每天及全程汇总。"""
    total_distance = 0.0
    total_duration = 0
    verified_days = []

    current_location: str | None = None
    for day in plan["days"]:
        places = [day["start"], *day.get("waypoints", []), day["end"]]
        # 起终点相同且无途经点是原地游览日，不调用驾车接口。
        legs = []
        for start, end in zip(places, places[1:]):
            if start.strip() == end.strip():
                continue
            leg = _route_leg(start, end, current_location)
            current_location = leg.pop("_end_location")
            legs.append(leg)
        distance = round(sum(leg["distance_km"] for leg in legs), 1)
        duration = sum(leg["duration_minutes"] for leg in legs)
        total_distance += distance
        total_duration += duration
        verified_days.append(
            {
                **day,
                "legs": legs,
                "distance_km": distance,
                "duration_minutes": duration,
            }
        )

    return {
        "trip_title": plan.get("trip_title", "自驾行程"),
        "day_count": len(verified_days),
        "total_distance_km": round(total_distance, 1),
        "total_duration_minutes": total_duration,
        "days": verified_days,
    }


def _daily_drive_limit(requirements: str) -> int | None:
    """从用户要求中提取每日驾驶上限，返回分钟数。"""
    match = re.search(
        r"(?:每天|每日).*?(?:不超过|最多|控制在|少于)\s*(\d+(?:\.\d+)?)\s*(?:小时|h)",
        requirements,
        re.IGNORECASE,
    )
    return round(float(match.group(1)) * 60) if match else None


def _limit_violations(
    verified: dict[str, Any], limit_minutes: int | None
) -> list[str]:
    if limit_minutes is None:
        return []
    return [
        f"第{day['day']}天实际驾驶{day['duration_minutes']}分钟"
        for day in verified["days"]
        if day["duration_minutes"] > limit_minutes
    ]


def format_verified_trip(verified: dict[str, Any]) -> str:
    """不依赖大模型，将高德核验结果格式化为可直接阅读的行程。"""

    def duration_text(minutes: int) -> str:
        hours, remaining = divmod(minutes, 60)
        return f"约 {hours} 小时 {remaining} 分钟"

    lines = [
        f"# {verified['trip_title']}",
        "",
        (
            f"共 {verified['day_count']} 天，全程约 "
            f"{verified['total_distance_km']:.1f} 公里，"
            f"预计驾驶 {duration_text(verified['total_duration_minutes'])}。"
        ),
    ]
    for day in verified["days"]:
        lines.extend(
            [
                "",
                f"## 第 {day['day']} 天：{day['start']} → {day['end']}",
                "",
                f"- 起点：{day['start']}",
                f"- 终点：{day['end']}",
                f"- 当日里程：约 {day['distance_km']:.1f} 公里",
                f"- 驾驶时长：{duration_text(day['duration_minutes'])}",
            ]
        )
        if day.get("waypoints"):
            lines.append(f"- 途经点：{'、'.join(day['waypoints'])}")
        lines.extend(["", "分段路线："])
        if day["legs"]:
            for leg in day["legs"]:
                roads = "、".join(leg["main_roads"]) or "以高德实时导航为准"
                lines.append(
                    f"- {leg['start']} → {leg['end']}：约 "
                    f"{leg['distance_km']:.1f} 公里，"
                    f"{duration_text(leg['duration_minutes'])}；主要道路：{roads}。"
                )
        else:
            lines.append("- 当天为原地游览，不安排长途驾驶。")
        if day.get("activities"):
            lines.extend(["", f"活动建议：{'；'.join(day['activities'])}"])
        if day.get("notes"):
            lines.extend(["", f"注意事项：{day['notes']}"])
    return "\n".join(lines)


def design_trip(origin: str, requirements: str) -> str:
    """执行“模型规划 -> 高德核验 -> 模型整理”的完整 Agent 流程。"""
    verified = _plan_and_verify(origin, requirements)
    return render_trip(verified, requirements)


def _plan_and_verify(
    origin: str,
    requirements: str,
    previous_trip: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """规划并核验；若违反每日驾驶上限，最多自动重排两次。"""
    limit_minutes = _daily_drive_limit(requirements)
    revision_base = previous_trip
    instruction = requirements

    for attempt in range(3):
        draft = create_draft(origin, instruction, revision_base)
        verified = verify_routes(draft)
        violations = _limit_violations(verified, limit_minutes)
        if not violations:
            return verified
        if attempt == 2:
            details = "、".join(violations)
            raise RuntimeError(f"连续调整后仍未满足每日驾驶上限：{details}")

        revision_base = verified
        details = "、".join(violations)
        instruction = (
            f"继续满足用户原始要求：{requirements}\n"
            f"高德核验发现以下超限：{details}。"
            f"每日驾驶必须不超过{limit_minutes}分钟。请增加天数、调整住宿地或删减绕行点，"
            "重新生成完整路线；不得用备注中的估算时间代替地图时间。"
        )

    raise RuntimeError("路线自动调整失败")


class TravelRouteAgent:
    """保存路线状态、支持用户连续微调的自驾路线 Agent。"""

    def __init__(self, origin: str) -> None:
        if not origin.strip():
            raise ValueError("起始点不能为空")
        self.origin = origin.strip()
        self.current_trip: dict[str, Any] | None = None
        self.requirement_history: list[str] = []

    def chat(self, user_message: str) -> str:
        """处理一轮规划或修改，并保存成功核验后的新版路线。"""
        message = user_message.strip()
        if not message:
            raise ValueError("路线要求不能为空")

        # 修改轮同时继承之前已明确提出的约束，避免模型遗忘每日时长等要求。
        combined_requirements = "\n".join([*self.requirement_history, message])
        verified = _plan_and_verify(
            self.origin,
            combined_requirements,
            self.current_trip,
        )
        # 只有模型规划和地图核验都成功后才替换当前版本。
        self.current_trip = verified
        self.requirement_history.append(message)
        return render_trip(verified, message)


def render_trip(verified: dict[str, Any], user_message: str) -> str:
    """优先由模型整理行程，模型不可用时输出本地格式化结果。"""
    try:
        return call_llm(
            [
                {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户本轮要求：{user_message}\n\n"
                        "以下是已经高德地图核验的最新行程数据：\n"
                        f"{json.dumps(verified, ensure_ascii=False, indent=2)}"
                    ),
                },
            ],
            temperature=0.4,
            reasoning_effort="low",
            timeout=120,
        )
    except Exception as exc:
        fallback = format_verified_trip(verified)
        return f"{fallback}\n\n> 大模型润色暂时不可用，已输出高德核验结果：{exc}"


def main() -> None:
    # Windows 控制台常默认为 GBK，模型输出 emoji 时会触发编码异常。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="大模型 + 高德地图自驾路线 Agent")
    parser.add_argument("--origin", help="自驾起始点")
    parser.add_argument("--requirements", help="目的地、天数、偏好等具体要求")
    args = parser.parse_args()

    origin = args.origin or input("请输入起始点：").strip()
    requirements = args.requirements or input("请输入具体要求：").strip()
    if not origin or not requirements:
        parser.error("起始点和具体要求不能为空")

    agent = TravelRouteAgent(origin)
    print("\n正在生成路线草案并调用高德地图核验，请稍候……\n")
    try:
        print(agent.chat(requirements))
    except Exception as exc:
        raise SystemExit(f"路线规划失败：{exc}") from exc

    print(
        "\n你可以继续输入修改要求，例如“第三天不去花湖”或“每天最多开4小时”。"
        "输入“完成”或“退出”结束。"
    )
    while True:
        try:
            adjustment = input("\n请微调路线：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n路线规划结束。")
            break
        if adjustment.lower() in {"完成", "退出", "exit", "quit", "q"}:
            print("路线规划结束。")
            break
        if not adjustment:
            continue
        print("\n正在根据意见调整并重新调用高德地图核验，请稍候……\n")
        try:
            print(agent.chat(adjustment))
        except Exception as exc:
            print(f"本轮调整失败，已保留上一版路线：{exc}")


if __name__ == "__main__":
    main()
